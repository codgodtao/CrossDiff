import torch
import torch.nn as nn

from model.guided_diffusion_modules.nn import gamma_embedding
import torch.nn.functional as F

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def Reverse(lst):
    return [ele for ele in reversed(lst)]


class ResblockUpOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockUpOne, self).__init__()

        self.conv20 = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.dense1 = Dense(embed_dim, channel_in)

        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


class ResblockDownOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockDownOne, self).__init__()

        self.conv20 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2, padding=1)
        self.dense1 = Dense(embed_dim, channel_in)  # 转换的形状为(batch_size,inter_dim)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)  # 输入的x每个元素都需要加上embed信息
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h

class Multi_branch_Unet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.
    输入为XT,MS 输出为PAN"""

    def __init__(self, channels=None, embed_dim=256, inter_dim=32, spectrun_num=4):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))

        self.conv1 = nn.Conv2d(1 + spectrun_num, 32, kernel_size=3, stride=1, padding=1)
        self.res3 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)

        self.down1 = ResblockDownOne(channels[1], channels[2], embed_dim, 32)  # 三个分支合并，通道数量*3

        self.down2 = ResblockDownOne(channels[2], channels[3], embed_dim, 32)

        self.up1 = ResblockUpOne(channels[3], channels[2], embed_dim, 32)

        self.dense2 = Dense(embed_dim, channels[2])
        self.groupnorm2 = nn.GroupNorm(32, num_channels=channels[2])
        self.up2 = nn.ConvTranspose2d(in_channels=channels[2] + channels[2], out_channels=channels[1], kernel_size=3,
                                      stride=2, output_padding=1, padding=1)

        self.dense3 = Dense(embed_dim, channels[1])
        self.groupnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.up3 = nn.ConvTranspose2d(in_channels=channels[1] + channels[1], out_channels=channels[0],
                                      kernel_size=3, stride=2, output_padding=1, padding=1)

        self.dense4 = Dense(embed_dim, channels[0])
        self.groupnorm4 = nn.GroupNorm(4, num_channels=channels[0])
        self.final1 = nn.Conv2d(in_channels=channels[0] + channels[0], out_channels=channels[0], kernel_size=3,
                                stride=1, padding=1)

        self.dense5 = Dense(embed_dim, channels[0])
        self.groupnorm5 = nn.GroupNorm(4, num_channels=channels[0])
        self.final2 = nn.Conv2d(channels[0] + spectrun_num + 1, spectrun_num, kernel_size=3, stride=1, padding=1)

        self.act = SiLU()

    def forward(self, x_t, t_input, cond, feat_need=False):  # 修改模型输入内容，修改为MS,PAN,NOISE三个部分，不能切片输入
        # Obtain the Gaussian random feature embedding for t
        t = t_input.view(-1, )
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理

        h1 = self.conv1(torch.cat([x_t, cond], dim=1))  # 32
        fe = [h1]
        # h2 = self.res1(h1, embed)  # 64
        h2 = self.res3(h1, embed)
        h3 = self.down1(h2, embed)  # 128
        h4 = self.down2(h3, embed)  # 256
        fe.append(h2)
        fe.append(h3)
        fe.append(h4)
        h = self.up1(h4, embed)  # 128
        fd = [h]
        h += self.dense2(embed)
        h = self.groupnorm2(h)
        h = self.act(h)
        h = self.up2(torch.cat([h, h3], dim=1))  # 64
        fd.append(h)
        h += self.dense3(embed)
        h = self.groupnorm3(h)
        h = self.act(h)
        h = self.up3(torch.cat([h, h2], dim=1))  # 32
        fd.append(h)
        h += self.dense4(embed)
        h = self.groupnorm4(h)
        h = self.act(h)
        h = self.final1(torch.cat([h, h1], dim=1))  # 32
        fd.append(h)
        h += self.dense5(embed)
        h = self.groupnorm5(h)
        h = self.act(h)
        h = self.final2(torch.cat([h, x_t, cond], dim=1))
        fd.append(h)
        if feat_need:
            return fe, Reverse(fd)
        else:
            return h

    # def forward(self, x_t):
    #     # Obtain the Gaussian random feature embedding for t
    #     cond = torch.randn((2, 1, 256, 256))
    #     t_input = torch.zeros(1)
    #     feat_need = False
    #     t = t_input.view(-1, )
    #     embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理
    #
    #     h1 = self.conv1(torch.cat([x_t, cond], dim=1))  # 32
    #     fe = [h1]
    #     # h2 = self.res1(h1, embed)  # 64
    #     h2 = self.res3(h1, embed)
    #     h3 = self.down1(h2, embed)  # 128
    #     h4 = self.down2(h3, embed)  # 256
    #     fe.append(h2)
    #     fe.append(h3)
    #     fe.append(h4)
    #     h = self.up1(h4, embed)  # 128
    #     fd = [h]
    #     h += self.dense2(embed)
    #     h = self.groupnorm2(h)
    #     h = self.act(h)
    #     h = self.up2(torch.cat([h, h3], dim=1))  # 64
    #     fd.append(h)
    #     h += self.dense3(embed)
    #     h = self.groupnorm3(h)
    #     h = self.act(h)
    #     h = self.up3(torch.cat([h, h2], dim=1))  # 32
    #     fd.append(h)
    #     h += self.dense4(embed)
    #     h = self.groupnorm4(h)
    #     h = self.act(h)
    #     h = self.final1(torch.cat([h, h1], dim=1))  # 32
    #     fd.append(h)
    #     h += self.dense5(embed)
    #     h = self.groupnorm5(h)
    #     h = self.act(h)
    #     h = self.final2(torch.cat([h, x_t, cond], dim=1))
    #     fd.append(h)
    #     if feat_need:
    #         return fe, Reverse(fd)
    #     else:
    #         return h


from torchsummary import summary

if __name__ == '__main__':
    model = Multi_branch_Unet(channels=[32, 64, 128, 256], spectrun_num=4)
    summary(model, (4, 256, 256))
