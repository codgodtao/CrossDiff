# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.QNR_LOSS import QNR
from core.dynamic_conv import Dynamic_conv2d
from core.mylib import sobel_gradient, channel_pooling, get_hp, get_lp
from model.easy_unet.se import ChannelSpatialSELayer
from data.util import ssim, get_gaussian_kernel


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.LeakyReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
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
        self.conv20 = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=3, stride=1, padding=1,
                                     bias=True)
        self.conv21 = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=3, stride=1, padding=1,
                                     bias=True)

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
            dim = self.channel_multiplier[i] * 2  #

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.channel_multiplier) - 1:
                dim_out = self.channel_multiplier[i + 1] * 2  #
                self.decoder.append(
                    AttentionBlock(dim=dim, dim_out=dim_out)
                )

        # Final classification head
        self.final = Dynamic_conv2d(dim_out, inner_channel, kernel_size=3, padding=2, stride=1)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2
        )
        self.final2 = Dynamic_conv2d(inner_channel * 2, spectral_num, kernel_size=3, padding=1, stride=1)
        self.relu = nn.LeakyReLU()
        self.up2 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=128 * 2,
                                      kernel_size=3, stride=2, output_padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64 * 2,
                                      kernel_size=3, stride=2, output_padding=1, bias=False)
        self.up4 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32 * 2,
                                      kernel_size=3, stride=2, output_padding=1, bias=False)
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

                diff = layer(torch.cat((f_A, f_B), dim=1))  # 改为加法看一下效果 并且A-B B-A试一下！
                if lvl != 0:
                    diff = diff + x
                lvl += 1
            # feature map的规律，通道数量两倍增加，size两倍降低
            else:  # 2倍上采样操作diff=32,x=64;diff=64,diff+x,x=128;diff=128,diff+x,x=256;diff=256,x+diff 3次操作,dim_out每次减掉一半
                diff = layer(diff)
                if lvl == 1:
                    x = self.up2(diff)
                elif lvl == 2:
                    x = self.up3(diff)
                elif lvl == 3:
                    x = self.up4(diff)
        # fusion result 卷积操作之后接一个norm操作，然后再激活函数
        result1 = self.final(x)
        result = self.backbone(self.relu(result1))  # 中间通道为64?加一个concat？
        result = self.final2(torch.cat([result, result1], dim=1))
        sr = torch.add(MS, result)
        return sr

    def spatial_loss_pangan(self, x_hat, pan):
        # channel mean
        out2pan = torch.mean(x_hat, dim=1).unsqueeze(1)
        pan_gradient_x, pan_gradient_y = sobel_gradient(pan)
        out2pan_gradient_x, out2pan_gradient_y = sobel_gradient(out2pan)
        loss_pan_out = self.loss_func(pan_gradient_x, out2pan_gradient_x) + \
                       self.loss_func(pan_gradient_y, out2pan_gradient_y)
        return loss_pan_out

    def spectral_loss_pangan(self, x_hat, ms):
        [b, c, h, w] = ms.shape
        x_hat_down = F.interpolate(x_hat, size=h, mode='bicubic')
        loss = self.loss_func(x_hat_down, ms)
        return loss

    def spatial_loss_ucgan(self, x_hat, pan):
        fake_pan = channel_pooling(x_hat, mode='max')
        loss = self.loss_func(get_hp(fake_pan), get_hp(pan))
        return loss

    def spectral_loss_ucgan(self, x_hat, ms):
        loss = self.loss_func(get_lp(x_hat), get_lp(ms))
        return loss

    def spa(self, x_hat, pan):
        fake_pan = channel_pooling(x_hat, mode='max')
        loss = self.loss_func_2(get_hp(fake_pan), get_hp(pan)) + (1 - ssim(fake_pan, pan, 5, 'mean', 1.))
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

    def train_step(self, x_in, feats_A, feats_B):
        PAN = x_in['PAN_0']  # 【0-1】
        MS = x_in['MS_0']  # [0-1]
        LR = x_in['LR_0']  # 【0-1】
        # GT = x_in['HR']
        sr = self(PAN, MS, feats_A, feats_B)  # 期望【0-1】
        # 分别约束空间和光谱信息！
        spatial_loss = self.spe1(sr, LR, 4) * 0.0001
        spectral_loss = self.spe2(sr, MS, 4) * 0.00001
        # spa_loss = self.spa(sr, PAN) * 0.0001
        qnr_loss = (1 - self.qnr(PAN, LR, sr)[0])

        # print(self.spe1(sr, LR, 4) * 0.0001, self.spe2(sr, MS, 4) * 0.00001, qnr_loss)  # 这些损失的取值范围如何，变化趋势如何？
        # loss = self.loss_func(sr, GT)
        # return loss
        return spectral_loss + spatial_loss + qnr_loss

    @torch.no_grad()
    def val_step(self, x_in, feats_A, feats_B):
        PAN = x_in['PAN_0']
        MS = x_in['MS_0']
        sr = self(PAN, MS, feats_A, feats_B)
        return sr

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
            self.loss_func_2 = nn.L1Loss().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
            self.loss_func_2 = nn.L1Loss().to(device)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    netCD = FusionNet(spectral_num=4,
                      loss_type="l2",
                      channel_multiplier=[32, 64, 128, 256],
                      time_steps=[50, 400],
                      inner_channel=32)
    print(netCD)
    feature_map_2 = torch.randn((8, 256, 30, 30))
    feature_map_3 = torch.randn((8, 128, 62, 62))
    feature_map_4 = torch.randn((8, 64, 126, 126))
    feature_map_5 = torch.randn((8, 32, 254, 254))
    feats_A = [feature_map_2, feature_map_3, feature_map_4, feature_map_5]
    feats_A = Reverse(feats_A)
    feats_B = feats_A.copy()
    # f_A = [feats_A, feats_A.copy(), feats_A.copy()]
    # f_B = [feats_B, feats_B.copy(), feats_B.copy()]
    f_A = [feats_A, feats_A.copy()]
    f_B = [feats_B, feats_B.copy()]
    PAN = torch.randn((8, 1, 256, 256))
    MS = torch.randn((8, 4, 256, 256))
    # x_in = {}
    # x_in['PAN_0'] = torch.randn((8, 1, 256, 256))
    # x_in['MS_0'] = torch.randn((8, 4, 256, 256))
    # x_in['HR'] = torch.randn((8, 4, 256, 256))
    # x_in['LR_0'] = torch.randn((8, 4, 64, 64))
    output = netCD(PAN, MS, f_A, f_B)
    # netCD.set_loss('cuda')
    # loss = netCD.train_step(x_in, f_A, f_B)
    # print(output.shape)  # torch.Size([8, 2, 256, 256])
