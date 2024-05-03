import torch
import data as Data
import argparse
import logging
import core.logger as Logger
from tensorboardX import SummaryWriter
import os.path as osp
import os
import numpy as np
import scipy.io as scio
import model as Model
from core.QNR_LOSS import QNR
import time

from core.unsupervised_metric.metrics import DRho

eps = torch.finfo(torch.float32).eps


def normlizaton(tensor, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    return tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/fusion_qb_full.json',
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
        best_qnr_val = 0.0
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
                qnr_val, d_lambda_val, d_s_val, d_rho = 0., 0., 0., 0.

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
                    qnr_, d_lambda, d_s = qnr(visuals['PAN_0'], visuals['LR_0'],
                                              result)  # [0-1] 结果仍然是torch，没有转为numpy
                    qnr_val += qnr_
                    d_lambda_val += d_lambda
                    d_s_val += d_s.data.squeeze().float().clamp_(0, 1).cpu().numpy()
                    scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx)}.mat'),
                                 {"sr": np.transpose(result[0] * 2047., (1, 2, 0)).detach().numpy()})  # H*W*C

                    result = np.transpose(result[0] * 2047., (1, 2, 0)).detach().numpy()
                    pan = np.transpose(visuals['PAN_0'][0] * 2047., (1, 2, 0)).detach().numpy()
                    d_rho += DRho(result, pan, sigma=4)  # 转为 H W C的numpy运算

                qnr_val = float(qnr_val / val_loader.__len__())
                d_lambda_val = float(d_lambda_val / val_loader.__len__())
                d_s_val = float(d_s_val / val_loader.__len__())
                d_rho = float(d_rho / val_loader.__len__())
                logger.info(qnr_val, d_rho, d_lambda_val, d_s_val)
                tb_logger.add_scalar('d_rho on validation data', d_rho, current_epoch)
                tb_logger.add_scalar('QNR on validation data', qnr_val, current_epoch)
                tb_logger.add_scalar('D_lambda on validation data', d_lambda_val, current_epoch)
                tb_logger.add_scalar('D_s on validation data', d_s_val, current_epoch)
            # save model
            if current_epoch % opt['train']['save_checkpoint_freq'] == 0:
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
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        qnr_val, d_lambda_val, d_s_val = 0., 0., 0.
        result_path = '{}/{}'.format(opt['path']
                                     ['results'], current_epoch)
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
            start_time = time.time()
            fusionNet.test()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            visuals = fusionNet.get_current_visuals()

            result = normlizaton(visuals['SR'])  # [0,1]
            qnr_, d_lambda, d_s = qnr(visuals['PAN_0'], visuals['LR_0'], result)
            qnr_val += qnr_
            d_lambda_val += d_lambda
            d_s_val += d_s
            scio.savemat(osp.join(result_path, f'output_mulExm_{str(idx)}.mat'),
                         {"sr": np.transpose(result[0] * 2047., (1, 2, 0)).detach().numpy()})  # H*W*C

        qnr_val = float(qnr_val / test_loader.__len__())
        d_lambda_val = float(d_lambda_val / test_loader.__len__())
        d_s_val = float(d_s_val / test_loader.__len__())
        tb_logger.add_scalar('QNR on validation data', qnr_val, current_epoch)
        tb_logger.add_scalar('D_lambda on validation data', d_lambda_val, current_epoch)
        tb_logger.add_scalar('D_s on validation data', d_s_val, current_epoch)
