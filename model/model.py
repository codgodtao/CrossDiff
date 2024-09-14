from collections import OrderedDict
import os

from torch.optim import lr_scheduler

from model.networks import *
import model.networks as networks
from model.base_model import BaseModel
from copy import deepcopy
import torch
from data.util import psnr_loss, ssim, sam

eps = torch.finfo(torch.float32).eps

logger = logging.getLogger('base')


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)  # 保留一个模型的备份，model的参数更新很快，我们只使用model的一部分参数，dacay是旧参数，1-decay是model的新参数
        self.module.eval()
        self.decay = decay
        self.device = device  # 如果设置device就会在不同的设备上执行ema
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DDPM(BaseModel):
    '''
        DDPM集成了diffusion network and unet
        主要用于调用diffusion model接口，进行业务层面的修改
    '''

    def __init__(self, opt, choice):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.optG = None
        self.scheduler = None
        self.choice = choice
        self.netG = self.set_device(networks.define_G(opt, choice))
        self.schedule_phase = None
        self.set_loss()

        if self.opt['phase'] == 'train':  # 如果有预训练模型，之前初始化的参数，这里初始化的scheduler和optG会被覆盖掉
            self.netG.train()
            optim_params = list(self.netG.parameters())  # 加载所有参数进行训练
            self.optG = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"], weight_decay=0)
            self.scheduler = lr_scheduler.StepLR(self.optG, step_size=1000,
                                                 gamma=0.5)
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def get_feats(self, t):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                fe_A, fd_A = self.netG.module.feats(self.data, t)
            else:
                fe_A, fd_A = self.netG.feats(self.data, t)
        self.netG.train()
        return fe_A, fd_A

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)  # 调用forward函数，计算扩散loss
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False, w=3.0):  # 传入的continous为True
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data, continous, w=w)
            else:
                self.SR = self.netG.super_resolution(  # SR是连续采样得到的结果
                    self.data, continous, w=w)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()  # 预测的MS
            out_dict['MS'] = self.data['MS'].detach().float().cpu()  # 上采样的MS，pan2ms的GT
            out_dict['PAN'] = self.data['PAN'].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']  # MS没有上采样的结果
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        # ema_path = os.path.join(
        #     self.opt['path']['checkpoint'], 'EMA_I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': self.scheduler.state_dict(),
                     'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        # 选择加载pan2ms,ms2pan两个不同的模型
        if self.choice == "pan2ms":
            load_path = self.opt['path']['pan2ms']
        else:
            load_path = self.opt['path']['ms2pan']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module

            network.load_state_dict(torch.load(  # strict set to False,ignore non-matching keys
                gen_path), strict=(self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                self.optG.load_state_dict(opt['optimizer'])
                self.scheduler.load_state_dict(opt['scheduler'])  # 在之前的预训练模型中不存在
                # optim_params = list(self.netG.parameters())  # 加载所有参数进行训练，和之前的优化器不一样
                # self.optG = torch.optim.Adam(
                #     optim_params, lr=self.opt['train']["optimizer"]["lr"])
                # self.scheduler = lr_scheduler.StepLR(self.optG, step_size=self.opt['train']['optimizer']['step_size'],
                #                                      gamma=self.opt['train']['optimizer']['gamma'])

                #
                #   尝试2，先跑几个epoch将新加的两层有个不错的结果（其他所有层不变），然后保存后整体模型进行微调得到结果
                #     optim_params = []
                #     for k, v in network.named_parameters():
                #         print(k, v.requires_grad)
                #         v.requires_grad = False
                #         if k in ['denoise_fn.downs.0.weight', 'denoise_fn.downs.0.bias',
                #                  'denoise_fn.final_conv.block.0.weight',
                #                  'denoise_fn.final_conv.block.0.bias',
                #                  'denoise_fn.final_conv.block.3.weight',
                #                  'denoise_fn.final_conv.block.3.bias']:
                #             v.requires_grad = True
                #             optim_params.append(v)
                #             logger.info(
                #                 'Params [{:s}] will optimize.'.format(k))
                #     self.optG = torch.optim.Adam(optim_params, self.opt['train']["optimizer"]["lr"])
                #
                #
                #
                #     尝试1：一部分微调，一部分大改
                #     find the parameters to optimize,分组训练，学习率区别开来
                # total = [name for name, param in network.named_parameters()]
                #
                # param_lx = [param for name, param in network.named_parameters()
                #             if name not in ['denoise_fn.downs.0.weight', 'denoise_fn.downs.0.bias',
                #                             'denoise_fn.final_conv.block.0.weight',
                #                             'denoise_fn.final_conv.block.0.bias',
                #                             'denoise_fn.final_conv.block.3.weight',
                #                             'denoise_fn.final_conv.block.3.bias']]
                # param_fi = [param for name, param in network.named_parameters()
                #             if name in ['denoise_fn.downs.0.weight', 'denoise_fn.downs.0.bias',
                #                         'denoise_fn.final_conv.block.0.weight',
                #                         'denoise_fn.final_conv.block.0.bias',
                #                         'denoise_fn.final_conv.block.3.weight',
                #                         'denoise_fn.final_conv.block.3.bias']]
                # learning_rate = self.opt['train']["optimizer"]["lr"]
                # # total = [param for name, param in network.named_parameters()]
                #
                # print("parameter group to train", len(total), len(param_lx), len(param_fi))
                # self.optG = torch.optim.Adam(
                #     [{'params': param_lx},
                #      {'params': param_fi,
                #       'lr': learning_rate * 100}], lr=learning_rate)
