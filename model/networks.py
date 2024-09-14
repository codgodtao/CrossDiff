import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from model.easy_unet import diffusion_pan2ms, diffusion_ms2pan
from model.fusion_modules import fusionnet_2 as fusionnet

logger = logging.getLogger('base')


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


def define_G(opt, choice):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'guide':
        from .guided_diffusion_modules import unet
        model = unet.UNet2(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            dropout=model_opt['unet']['dropout']
        )
    elif model_opt['which_model_G'] == 'sr3':
        from .sr import unet
        if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
            model_opt['unet']['norm_groups'] = 32
        model = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
    else:
        if choice == "ms2pan":
            from .easy_unet import ms_pan_v2 as ms_pan
            model = ms_pan.Multi_branch_Unet(channels=model_opt['unet']['channel_multiplier'],
                                             spectrun_num=model_opt['unet']['in_channel'])
            netG = diffusion_ms2pan.GaussianDiffusion(
                denoise_fn=model,  # denoise_fn is as unet, so
                loss_type=model_opt['diffusion']['loss_type'],  # L1 or L2
                conditional=model_opt['diffusion']['conditional'])  # True or False
        else:
            from .easy_unet import pan_ms_v2 as pan_ms
            model = pan_ms.Multi_branch_Unet(channels=model_opt['unet']['channel_multiplier'],
                                             spectrun_num=model_opt['unet']['in_channel'])
            netG = diffusion_pan2ms.GaussianDiffusion(
                denoise_fn=model,  # denoise_fn is as unet, so
                loss_type=model_opt['diffusion']['loss_type'],  # L1 or L2
                conditional=model_opt['diffusion']['conditional'])  # True or False
    if opt['phase'] == 'train':
        init_weights(netG, init_type=model_opt['init_type'])
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        logger.info("distributed_training")
        netG = nn.DataParallel(netG)
    return netG


def define_Fusion(opt):
    model_opt = opt['model']
    model_fu_opt = opt['model_fu']
    netG = fusionnet.FusionNet(
        spectral_num=model_fu_opt['spectral_num'],
        loss_type=model_opt['diffusion']['loss_type'],
        channel_multiplier=model_fu_opt['channel_multiplier'],
        time_steps=model_fu_opt["t"],
        inner_channel=model_fu_opt['inner_channel']

    )
    if opt['fusion_phase'] == 'train':
        init_weights(netG, init_type=model_opt['init_type'])
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        logger.info("distributed_training")
        netG = nn.DataParallel(netG)
    return netG
