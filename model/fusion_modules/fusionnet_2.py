# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from core.QNR_LOSS import QNR, up_sample, QIndex_torch
from core.mylib import sobel_gradient, channel_pooling, get_hp, get_lp, get_hp_gussain
from core.unsupervised_metric.metrics import SAM, SAM_torch
from data.util import ssim, get_gaussian_kernel
from model.easy_unet.se import ChannelSpatialSELayer


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.LeakyReLU(),
            # nn.Conv2d(dim_out, dim_out, 3, padding=1)
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)  # 通道数量减半
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        if len(time_steps) > 1:
            self.block = nn.Sequential(
                nn.Conv2d(dim * len(time_steps), dim_out, 3, padding=1),
                nn.LeakyReLU()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim_out, 3, padding=1),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.block(x)


def Reverse(lst):
    return [ele for ele in reversed(lst)]


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class FusionNet(nn.Module):
    def __init__(self, spectral_num=4, inner_channel=32,
                 channel_multiplier=None, time_steps=None, loss_type='l1'):
        super(FusionNet, self).__init__()

        self.loss_type = loss_type

        # Define the parameters of the change detection head
        channel_multiplier.sort(reverse=True)
        self.channel_multiplier = channel_multiplier  # [256,128,64,32]
        self.time_steps = time_steps

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.channel_multiplier)):
            dim = self.channel_multiplier[i] * 2
            self.decoder.append(  # 用于接收所有timestep下的输入
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )
            if i != len(self.channel_multiplier) - 1:
                dim_out = self.channel_multiplier[i + 1] * 2  #
                self.decoder.append(
                    AttentionBlock(dim=dim, dim_out=dim_out)
                )
        self.final = nn.Conv2d(dim_out, inner_channel, kernel_size=3, padding=1, stride=1)
        self.final2 = nn.Conv2d(inner_channel, spectral_num, kernel_size=3, padding=1, stride=1)
        self.act = nn.LeakyReLU()
        self.qnr = QNR()

    def forward(self, PAN, MS, feats_A, feats_B):
        # encoder
        lvl = 0
        for layer in self.decoder:
            if isinstance(layer, Block):  # 4次操作
                f_A = feats_A[0][3 - lvl]  # 第一个time step，每个step有4个level的特征
                f_B = feats_B[0][3 - lvl]
                for i in range(1, len(self.time_steps)):  # PAN-MS
                    f_A = torch.cat((f_A, feats_A[i][3 - lvl]), dim=1)  # 多个time stamp的特征在通道上连接起来，送入layer层中
                    f_B = torch.cat((f_B, feats_B[i][3 - lvl]), dim=1)  # 使用卷积层将通道数量再次降低，由于本文只是用一个time,这里没有实际用途
                diff = layer(torch.cat((f_A, f_B), dim=1))
                if lvl != 0:
                    diff = diff + x
                lvl += 1
            else:
                diff = layer(diff)
                x = up_sample(diff, r=2)

        result1 = self.act(self.final(x))
        # result1 = self.backbone(result1)
        result = self.final2(result1)
        sr = torch.add(MS, result)
        return sr

    # def forward(self, MS):
    #     feature_map_2 = torch.randn((2, 128, 32, 32))
    #     feature_map_3 = torch.randn((2, 64, 64, 64))
    #     feature_map_4 = torch.randn((2, 32, 128, 128))
    #     feature_map_5 = torch.randn((2, 16, 256, 256))
    #     feats_A = [feature_map_2, feature_map_3, feature_map_4, feature_map_5]
    #     feats_A = Reverse(feats_A)
    #     feats_B = feats_A.copy()
    #     lvl = 0
    #     for layer in self.decoder:
    #         if isinstance(layer, Block):  # 4次操作
    #             f_A = feats_A[3 - lvl]  # 第一个time step，每个step有4个level的特征
    #             f_B = feats_B[3 - lvl]
    #             diff = layer(torch.cat((f_A, f_B), dim=1))
    #             if lvl != 0:
    #                 diff = diff + x
    #             lvl += 1
    #         else:
    #             diff = layer(diff)
    #             x = up_sample(diff, r=2)
    #     result1 = self.act(self.final(x))
    #     result = self.final2(result1)
    #     sr = torch.add(MS, result)
    #     return sr

    def spa(self, x_hat, pan):
        "from supervised-unsupervised combined network"
        fake_pan = channel_pooling(x_hat, mode='avg')
        fake_pan = get_hp_gussain(fake_pan, 1)
        pan = get_hp_gussain(pan, 1)
        loss = self.loss_func(fake_pan, pan) + (
                1 - ssim(fake_pan, pan, 5, 'mean', 1.))
        return loss

    def spe1(self, x_hat, lr, spec_num):
        [b, c, h, w] = lr.shape
        guassian = get_gaussian_kernel(5, 2, spec_num).cuda()
        x_hat = guassian(x_hat)
        x_hat_down = F.interpolate(x_hat, size=h, mode='area')
        return self.loss_func(x_hat_down, lr) + (1 - ssim(x_hat_down, lr, 5, 'mean', 1.))

    def spe2(self, x_hat, ms, spec_num):
        x_hat_guass = get_gaussian_kernel(5, 2, spec_num).cuda()
        x_hat = x_hat_guass(x_hat)
        return self.loss_func(x_hat, ms) + (1 - ssim(x_hat, ms, 5, 'mean', 1.))

    def spe_Texture_Attention(self, x_hat, lr, spec_num):
        [b, c, h, w] = lr.shape
        guassian = get_gaussian_kernel(5, 2, spec_num).cuda()
        x_hat = guassian(x_hat)
        x_hat_down = F.interpolate(x_hat, size=h, mode='area')
        return SAM_torch(x_hat_down, lr)

    def Spatial_Texture_Attention(self, x_hat, pan):
        "Unsupervised Pansharpening Method Using Residual Network With Spatial Texture Attention"
        # 将pan从 B C H W拓展
        [b, c, h, w] = x_hat.shape
        pan = pan.repeat(1, c, 1, 1)
        return 1 - QIndex_torch(x_hat, pan)

    def Spatial_MUN_GAN(self, x_hat, pan):
        "Mun-GAN: A Multiscale Unsupervised Network for Remote Sensing Image Pansharpening"

        return

    def train_step(self, x_in, feats_A, feats_B, current_epoch):
        PAN = x_in['PAN_0']  # 【0-1】
        MS = x_in['MS_0']
        LR = x_in['LR_0']  # 【0-1】
        GT = x_in['HR']
        sr = self(PAN, MS, feats_A, feats_B)  # 期望【0-1】
        # spectral_loss_2 = self.spe2(sr, MS, 4) / 100
        # qnr_loss = (1 - self.qnr(PAN, LR, sr)[0])
        # spa_loss = self.spa(sr, PAN) / 100
        # print(qnr_loss.data, spectral_loss_2.data, spa_loss.data)
        # return spa_loss + spectral_loss_2 + qnr_loss
        return self.loss_func(sr, GT)

    @torch.no_grad()
    def val_step(self, x_in, feats_A, feats_B):
        PAN = x_in['PAN_0']
        MS = x_in['MS_0']
        sr = self(PAN, MS, feats_A, feats_B)
        return sr

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    netCD = FusionNet(spectral_num=4,
                      loss_type="l1",
                      channel_multiplier=[32, 64, 128, 256],
                      time_steps=[50],
                      inner_channel=32)
    feature_map_2 = torch.randn((2, 256, 32, 32))
    feature_map_3 = torch.randn((2, 128, 64, 64))
    feature_map_4 = torch.randn((2, 64, 128, 128))
    feature_map_5 = torch.randn((2, 32, 256, 256))
    feats_A = [feature_map_2, feature_map_3, feature_map_4, feature_map_5]
    feats_A = Reverse(feats_A)
    feats_B = feats_A.copy()
    pan = torch.randn(1, 1, 256, 256)
    ms = torch.randn(1, 4, 256, 256)
    print(netCD.forward(pan, ms, [feats_A], [feats_B]))
    # summary(netCD, (4, 256, 256))
    # pre = time.time()
    # netCD(torch.randn(1, 4, 256, 256))
    # after = time.time()
    # print(after - pre)
