import copy
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np
import scipy.io as scio
from data.util import psnr_loss, ssim, sam

eps = torch.finfo(torch.float32).eps


def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""

    N, C, H, W = img_real.shape
    means_real = img_real.reshape(N, C, -1).mean(dim=-1)
    mses = ((img_fake - img_real) ** 2).reshape(N, C, -1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS

    return 100 / scale * torch.sqrt((mses / (means_real ** 2 + eps)).mean())


def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
            eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)


def normlization(data, min_max=(-1, 1)):
    size = data.shape[-1]
    data = data.squeeze().float().cpu().clamp_(*min_max)
    data = (data - min_max[0]) / \
           (min_max[1] - min_max[0])
    return data.reshape((1, size, size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512_band4_linux.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(opt['info'])
    logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt, opt['choice'])
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epochs = opt['train']['n_epochs']

    diffusion.set_new_noise_schedule(  # 初始化扩散模型需要的，无需学习的参数
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    logger.info('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True, w=3.0)  # 传入的continues为True,一次只传入一个sample进行串行采样
        visuals = diffusion.get_current_visuals()

        # 保存需要的sr结果
        # img_scale = 2047.0
        # result = normlization(visuals['SR'][-1])
        # scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx - 1)}.mat'),
        #              {"sr": np.transpose(result * img_scale, (1, 2, 0)).detach().numpy()})  # H*W*C

        sr_img = visuals['SR']  # uint8
        MS_img = Metrics.tensor2img_4C(visuals['MS'])  # uint8
        PAN_img = Metrics.tensor2img_4C(visuals['PAN'])  # uint8
        Metrics.save_img(
                MS_img, '{}/{}_MS.png'.format(result_path,  idx))
        Metrics.save_img(
            PAN_img, '{}/{}_PAN.png'.format(result_path, idx))
        sample_num = sr_img.shape[0]
        for iter in range(0, sample_num):
            Metrics.save_img(
                Metrics.tensor2img_4C(sr_img[iter]),
                '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            img_scale = 2047.0
            result = normlization(sr_img[iter])
            scio.savemat('{}/{}_{}_sr_{}.mat'.format(result_path, current_step, idx, iter),
                         {"sr": np.transpose(result * img_scale, (1, 2, 0)).detach().numpy()})  # H*W*C
