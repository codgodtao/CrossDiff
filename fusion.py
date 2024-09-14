import torch
import data as Data
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os.path as osp
import os
import numpy as np
import scipy.io as scio
import model as Model
from core.QNR_LOSS import QNR
from data.util import get_metrics_reduced, psnr_loss

eps = torch.finfo(torch.float32).eps


def normlizaton(tensor, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    return tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_wv4.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
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
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # checkpoint pan2ms ms2pan for tow diffusion models
    ms_pan_model = Model.create_model(opt, "ms2pan")
    pan_ms_model = Model.create_model(opt, "pan2ms")

    ms_pan_model.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    pan_ms_model.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # fusion_model
    fusionNet = Model.create_fusionNet(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = fusionNet.begin_step
    current_epoch = fusionNet.begin_epoch
    n_epochs = opt['train']['n_epochs']
    qnr = QNR()

    if opt['fusion_phase'] == 'train':
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                current_epoch, current_step))
        best_psnr_val = 0.0
        while current_epoch < n_epochs:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += opt['datasets']['train']['batch_size']
                if current_epoch > n_epochs:
                    break
                pan_ms_model.feed_data(train_data)
                ms_pan_model.feed_data(train_data)
                f_pan2ms = []  # 双层列表，三个time,每个time内部是一组feature map
                f_ms2pan = []
                for t in opt['model_fu']['t']:
                    # 每次获取一个时间步上的feature map，然后送入列表整合就行
                    fe_pan2ms, fd_pan2ms = pan_ms_model.get_feats(t)
                    fe_ms2pan, fd_ms2pan = ms_pan_model.get_feats(t)
                    if opt['model_fu']['feat_type'] == "enc":
                        f_pan2ms.append(fe_pan2ms)
                        f_ms2pan.append(fe_ms2pan)
                        del fd_ms2pan, fd_pan2ms
                    else:
                        f_pan2ms.append(fd_pan2ms)
                        f_ms2pan.append(fd_ms2pan)
                        del fe_pan2ms, fe_ms2pan

                fusionNet.feed_data(train_data, f_pan2ms, f_ms2pan)
                fusionNet.optimize_parameters(current_epoch)
                # log
                if current_epoch % opt['train']['print_freq'] == 0:
                    logs = fusionNet.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

            fusionNet.scheduler.step()
            tb_logger.add_scalar("lr", fusionNet.scheduler.get_last_lr(), current_epoch)

            if current_epoch % opt['train']['val_freq'] == 0:
                psnr_val = 0.
                result_path = '{}/{}'.format(opt['path']
                                             ['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)
                for idx, val_data in enumerate(val_loader):
                    pan_ms_model.feed_data(val_data)
                    ms_pan_model.feed_data(val_data)
                    f_pan2ms = []  # 双层列表，三个time,每个time内部是一组feature map
                    f_ms2pan = []
                    for t in opt['model_fu']['t']:
                        # 由于验证阶段一层只有一个batch，因此存在一些特征提取上的问题
                        fe_pan2ms, fd_pan2ms = pan_ms_model.get_feats(t)
                        fe_ms2pan, fd_ms2pan = ms_pan_model.get_feats(t)
                        if opt['model_fu']['feat_type'] == "enc":
                            f_pan2ms.append(fe_pan2ms)
                            f_ms2pan.append(fe_ms2pan)
                            del fd_ms2pan, fd_pan2ms
                        else:
                            f_pan2ms.append(fd_pan2ms)
                            f_ms2pan.append(fd_ms2pan)
                            del fe_pan2ms, fe_ms2pan

                    fusionNet.feed_data(val_data, f_pan2ms, f_ms2pan)
                    fusionNet.test()
                    visuals = fusionNet.get_current_visuals()

                    result = normlizaton(visuals['SR'])  # [0,1]
                    img_scale = 2047.0
                    scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx)}.mat'),
                                 {"sr": np.transpose(result[0] * img_scale, (1, 2, 0)).detach().numpy()})  # H*W*C
                    psnr_val += psnr_loss(result, visuals['HR'], 1.)

                # qnr_val = float(qnr_val / val_loader.__len__())
                psnr_val = float(psnr_val / val_loader.__len__())
                logger.info(psnr_val)
                tb_logger.add_scalar('psnr on validation data', psnr_val, current_epoch)
            # save model
            if current_epoch % opt['train']['save_checkpoint_freq'] == 0:
                if best_psnr_val < psnr_val:
                    best_psnr_val = psnr_val
                    logger.info('Saving models and training states.')
                    fusionNet.save_network(current_epoch, current_step)

        logger.info('End of training.')
    else:
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'test':
                test_set = Data.create_dataset(dataset_opt, phase)
                test_loader = Data.create_dataloader(
                    test_set, dataset_opt, phase)
        logger.info('Begin Model Evaluation.')
        metrics = torch.zeros(5, val_loader.__len__())
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        qnr_val, D_lambda, D_s = 0., 0., 0.
        for idx, val_data in enumerate(test_loader):
            pan_ms_model.feed_data(val_data)
            ms_pan_model.feed_data(val_data)
            f_pan2ms = []  # 双层列表，三个time,每个time内部是一组feature map
            f_ms2pan = []
            for t in opt['model_fu']['t']:
                # 由于验证阶段一层只有一个batch，因此存在一些特征提取上的问题
                fe_pan2ms, fd_pan2ms = pan_ms_model.get_feats(t)
                fe_ms2pan, fd_ms2pan = ms_pan_model.get_feats(t)
                if opt['model_fu']['feat_type'] == "enc":
                    f_pan2ms.append(fe_pan2ms)
                    f_ms2pan.append(fe_ms2pan)
                    del fd_ms2pan, fd_pan2ms
                else:
                    f_pan2ms.append(fd_pan2ms)
                    f_ms2pan.append(fd_ms2pan)
                    del fe_pan2ms, fe_ms2pan

            fusionNet.feed_data(val_data, f_pan2ms, f_ms2pan)
            fusionNet.test()
            visuals = fusionNet.get_current_visuals()
            result = normlizaton(visuals['SR'])
            qnr_val_temp, D_lambda_temp, D_s_temp = qnr(visuals['PAN_0'], visuals['LR_0'], result)
            qnr_val += qnr_val_temp
            D_lambda += D_lambda_temp
            D_s += D_s_temp

            metrics[:, idx] = torch.Tensor(get_metrics_reduced(result, visuals['HR']))
            img_scale = 2047.0
            scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx)}.mat'),
                         {"sr": np.transpose(result[0] * img_scale, (1, 2, 0)).detach().numpy()})  # H*W*C

        for i in range(5):
            print(torch.mean(metrics[i, :]))
        # sr_img = Metrics.tensor2img_4C(visuals['SR'], min_max=(0, 1))  # uint8
        # GT_img = Metrics.tensor2img_4C(visuals['HR'], min_max=(0, 1))  # uint8
        # Metrics.save_img(sr_img,
        #                  '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
        # Metrics.save_img(
        #     GT_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
        qnr_val = float(qnr_val / test_loader.__len__())
        D_lambda = float(D_lambda / test_loader.__len__())
        D_s = float(D_s / test_loader.__len__())
        print(qnr_val, D_lambda, D_s)
