import os
from collections import OrderedDict
from torch.optim import lr_scheduler
from model.networks import *
import model.networks as networks
from model.base_model import BaseModel

eps = torch.finfo(torch.float32).eps
logger = logging.getLogger('base')


class fusionModel(BaseModel):

    def __init__(self, opt):
        super(fusionModel, self).__init__(opt)  # load opt as self.opt
        # define network and load pretrained models
        self.optG = None
        self.scheduler = None
        self.model = self.set_device(networks.define_Fusion(opt))
        self.set_loss()

        if self.opt['fusion_phase'] == 'train':  # 加载初始化后的模型，如果有预训练模型，这一步就会直接被覆盖掉
            self.model.train()
            optim_params = list(self.model.parameters())  # 加载所有参数进行训练
            self.optG = torch.optim.AdamW(
                optim_params, lr=opt['train']["optimizer"]["lr"], weight_decay=0)
            self.scheduler = lr_scheduler.StepLR(self.optG, step_size=100,
                                                 gamma=0.5)
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data, f_pan2ms, f_ms2pan):
        self.data = self.set_device(data)
        self.f_pan2ms = f_pan2ms
        self.f_ms2pan = f_ms2pan

    def optimize_parameters(self, current_epoch):
        self.optG.zero_grad()
        if isinstance(self.model, nn.DataParallel):
            logger.info("optimize_parameters in distributed training methods")
            # l_pix = self.model.module.train_step(self.data)
            l_pix = self.model.module.train_step(self.data, self.f_pan2ms, self.f_ms2pan,
                                                 current_epoch)  # 调用forward函数，计算扩散loss
        else:
            # l_pix = self.model.train_step(self.data)
            l_pix = self.model.train_step(self.data, self.f_pan2ms, self.f_ms2pan, current_epoch)
        # need to average in multi-gpu
        # b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()
        l_pix.backward()
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):  # 传入的continous为True
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, nn.DataParallel):
                self.SR = self.model.module.val_step(
                    self.data, self.f_pan2ms, self.f_ms2pan)
                # self.SR = self.model.module.val_step(
                #     self.data)
            else:
                self.SR = self.model.val_step(  # SR是连续采样得到的结果
                    self.data, self.f_pan2ms, self.f_ms2pan)
                # self.SR = self.model.module.val_step(
                #     self.data)
        self.model.train()

    def set_loss(self):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_loss(self.device)
        else:
            self.model.set_loss(self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['PAN'] = self.data['PAN'].float().cpu()  # [-1,1]用于特征提取
        out_dict['MS'] = self.data['MS'].detach().float().cpu()  # [-1,1]用于特征提取
        out_dict['LR'] = self.data['LR'].detach().float().cpu()  # [-1,1]

        out_dict['HR'] = self.data['HR'].detach().float().cpu()  # [0,1]模拟实验上的GT
        out_dict['SR'] = self.SR.detach().float().cpu()  # 预测的MS [0-1]
        out_dict['PAN_0'] = self.data['PAN_0'].detach().float().cpu()  # [0,1]模拟实验上的GT
        out_dict['MS_0'] = self.data['MS_0'].detach().float().cpu()  # [0,1]模拟实验上的GT
        out_dict['LR_0'] = self.data['LR_0'].detach().float().cpu()  # [0,1]模拟实验上的GT
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                             self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        # 加载的是最终的融合头
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.model
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
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.model
            if isinstance(self.model, nn.DataParallel):
                network = network.module

            network.load_state_dict(torch.load(  # strict set to False,ignore non-matching keys
                gen_path), strict=(self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                self.optG.load_state_dict(opt['optimizer'])
                self.scheduler.load_state_dict(opt['scheduler'])  # 在之前的预训练模型中不存在


import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchsummary import summary

if __name__ == '__main__':
    model = fusionModel().cuda()
    summary(model, input_size=(4, 64, 64))
    x = torch.randn((1, 4, 64, 64)).cuda()
    f1 = FlopCountAnalysis(model, x)
    print(f1.total())
