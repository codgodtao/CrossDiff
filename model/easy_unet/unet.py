import torch
import torch.nn as nn

from model.guided_diffusion_modules.nn import gamma_embedding


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ResblockUp(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1, num_group2):
        super(ResblockUp, self).__init__()

        self.conv20 = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2,
                                         output_padding=1
                                         )
        self.conv21 = nn.ConvTranspose2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, stride=1,
                                         padding=1)
        self.dense1 = Dense(embed_dim, channel_in)
        self.dense2 = Dense(embed_dim, channel_out)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.groupnorm2 = nn.GroupNorm(num_group2, num_channels=channel_out)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)

        h += self.dense2(embed)
        h = self.act(self.groupnorm2(h))
        h = self.conv21(h)

        return h


class ResblockDown(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1, num_group2):
        super(ResblockDown, self).__init__()

        self.conv20 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2)
        self.conv21 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, stride=1, padding=1)
        self.dense1 = Dense(embed_dim, channel_in)
        self.dense2 = Dense(embed_dim, channel_out)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.groupnorm2 = nn.GroupNorm(num_group2, num_channels=channel_out)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)

        h += self.dense2(embed)
        h = self.act(self.groupnorm2(h))
        h = self.conv21(h)

        return h


class Resblock(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1, num_group2):
        super(Resblock, self).__init__()

        self.conv20 = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1)
        self.conv21 = nn.ConvTranspose2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, stride=1,
                                         padding=1)
        self.dense1 = Dense(embed_dim, channel_in)
        self.dense2 = Dense(embed_dim, channel_out)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.groupnorm2 = nn.GroupNorm(num_group2, num_channels=channel_out)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)

        h += self.dense2(embed)
        h = self.act(self.groupnorm2(h))
        h = self.conv21(h)

        return h


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# class UNet(nn.Module):
#     def __init__(self, channels=None, embed_dim=256, inter_dim=32):
#         # 有一个基本的思路，既然可以利用编码信息，可以讲PAN和MS分别用过三层卷积提取特征，然后合并其中的信息，做到分别获取两种信息，从通道维度更新即可
#         super().__init__()
#         if channels is None:
#             channels = [32, 64, 128, 256]
#         self.inter_dim = inter_dim
#         self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))
#         self.conv1 = nn.Conv2d(9, channels[0], 3, stride=1)
#
#         self.res1 = ResblockDown(channels[0], channels[1], embed_dim, 4, 32)
#         self.res2 = ResblockDown(channels[1], channels[2], embed_dim, 32, 32)
#         self.res3 = ResblockDown(channels[2], channels[3], embed_dim, 32, 32)
#
#         self.res4 = ResblockUp(channels[3], channels[2], embed_dim, 32, 32)
#         self.res5 = ResblockUp(channels[2] + channels[2], channels[1], embed_dim, 32, 32)
#         self.res6 = ResblockUp(channels[1] + channels[1], channels[0], embed_dim, 32, 32)
#
#         self.final1 = Resblock(channels[0] + channels[0], 9, embed_dim, 32, 3)
#         self.final2 = nn.Conv2d(9 + 9, 4, kernel_size=3, stride=1, padding=1)
#         self.dense8 = Dense(embed_dim, 9)
#         self.tgnorm1 = nn.GroupNorm(3, 9)
#         self.act = lambda x: x * torch.sigmoid(x)
#
#     def forward(self, x, t):
#         t = t.view(-1, )
#         embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理
#         # Encoding path
#         h1 = self.conv1(x)  # 32*62*62
#
#         h2 = self.res1(h1, embed)  # 64*30*30
#         h3 = self.res2(h2, embed)  # 128*14*14
#         h4 = self.res3(h3, embed)  # 256*6*6
#         h = self.res4(h4, embed)  # 128*14*14
#         h = self.res5(torch.cat([h, h3], dim=1), embed)  # 64*30*30
#         h = self.res6(torch.cat([h, h2], dim=1), embed)  # 32*62*62
#         h = self.final1(torch.cat([h, h1], dim=1), embed)  # 9*64*64
#
#         h += self.dense8(embed)
#         h = self.tgnorm1(h)
#         h = self.act(h)
#         h = self.final2(torch.cat([h, x], dim=1))  # 最后一个输出层我加了一个激活层控制
#
#         return h


class UNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=None, embed_dim=256, inter_dim=32):
        super().__init__()
        # Gaussian random feature embedding layer for time
        # self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
        #                            nn.Linear(embed_dim, embed_dim))
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim
        # self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim), SiLU(), nn.Linear(embed_dim, embed_dim))
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(8, channels[0], 3, stride=1)
        self.dense1 = Dense(embed_dim, channels[0])  # 将时间编码信息在通道维度上修改，从而在每一个unet模块上都可以使用
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, output_padding=1)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 32, 3, stride=1)
        self.dense8 = Dense(embed_dim, 32)
        self.tgnorm1 = nn.GroupNorm(4, 32)

        self.final_conv = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        # The relu activation function or sigmod activation etc
        self.act = lambda x: x * torch.sigmoid(x)
        # self.act = nn.ReLU()
        # self.leaklyRelu = nn.LeakyReLU()

    def forward(self, MS, PAN, x, t):
        # Obtain the Gaussian random feature embedding for t
        t = t.view(-1, )
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理
        # Encoding path
        pan_concat = PAN.repeat(1, 4, 1, 1)  # Bsx8x64x64
        condition = torch.sub(pan_concat, MS)  # Bsx8x64x64
        input = torch.cat([condition, x], dim=1)
        h1 = self.conv1(input)

        ## Incorporate information from t
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)

        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)

        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)

        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # Decoding path
        h = self.tconv4(h4)

        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))

        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))

        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))  # 后面的都是新添加的内容

        h += self.dense8(embed)
        h = self.tgnorm1(h)
        h = self.act(h)
        h = self.final_conv(torch.cat([h, input], dim=1))

        return h


class ResblockUpOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockUpOne, self).__init__()

        self.conv20 = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2,
                                         output_padding=1
                                         )
        self.dense1 = Dense(embed_dim, channel_in)

        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        # self.groupnorm1 = nn.InstanceNorm2d(channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


class ResblockDownOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockDownOne, self).__init__()

        self.conv20 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2)
        self.dense1 = Dense(embed_dim, channel_in)  # 转换的形状为(batch_size,inter_dim)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        # self.groupnorm1 = nn.InstanceNorm2d(channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)  # 输入的x每个元素都需要加上embed信息
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


class ResblockOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockOne, self).__init__()

        self.conv20 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1, padding=1)
        self.dense1 = Dense(embed_dim, channel_in)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)

        return h


# class Multi_branch_Unet(nn.Module):
#     """A time-dependent score-based model built upon U-Net architecture."""
#
#     def __init__(self, channels=None, embed_dim=256, inter_dim=32):
#         super().__init__()
#         if channels is None:
#             channels = [32, 64, 128, 256]
#         self.inter_dim = inter_dim
#         self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))
#
#         self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
#         self.conv3 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
#
#         self.res1 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)
#
#         self.res2 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)
#
#         self.res3 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)
#
#         self.res4 = nn.Conv2d(channels[1] * 3, channels[1], 3, 1, padding=1)
#
#         self.down1 = ResblockDownOne(channels[1], channels[2], embed_dim, 32)  # 三个分支合并，通道数量*3
#
#         self.down2 = ResblockDownOne(channels[2], channels[3], embed_dim, 32)
#
#         self.up1 = ResblockUpOne(channels[3], channels[2], embed_dim, 32)
#
#         self.dense2 = Dense(embed_dim, channels[2])
#         self.groupnorm2 = nn.GroupNorm(32, num_channels=channels[2])
#         # self.groupnorm2 = nn.InstanceNorm2d(channels[2])
#         self.up2 = nn.ConvTranspose2d(in_channels=channels[2] + channels[2], out_channels=channels[1], kernel_size=3,
#                                       stride=2, output_padding=1)
#
#         self.dense3 = Dense(embed_dim, channels[1])
#         self.groupnorm3 = nn.GroupNorm(32, num_channels=channels[1])
#         # self.groupnorm3 = nn.InstanceNorm2d(channels[1])
#         self.up3 = nn.ConvTranspose2d(in_channels=channels[1] + channels[1], out_channels=channels[0],
#                                       kernel_size=3, stride=2, output_padding=1)
#
#         self.dense4 = Dense(embed_dim, channels[0])
#         self.groupnorm4 = nn.GroupNorm(4, num_channels=channels[0])
#         # self.groupnorm4 = nn.InstanceNorm2d(channels[0])
#         self.final1 = nn.ConvTranspose2d(in_channels=channels[0] + channels[0] * 3,
#                                          out_channels=channels[0], kernel_size=3, stride=1)
#
#         self.dense5 = Dense(embed_dim, channels[0])
#         self.groupnorm5 = nn.GroupNorm(4, num_channels=channels[0])
#         # self.groupnorm5 = nn.InstanceNorm2d(channels[0])
#         self.final2 = nn.Conv2d(channels[0], 4, kernel_size=3, stride=1, padding=1)
#         self.act = SiLU()
#
#     def forward(self, MS, PAN, x, t):  # 修改模型输入内容，修改为MS,PAN,NOISE三个部分，不能切片输入
#         # Obtain the Gaussian random feature embedding for t
#         t = t.view(-1, )
#         embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理
#
#         h1 = self.conv1(MS)  # 32
#         h2 = self.conv2(PAN)
#         h3 = self.conv3(x)
#
#         h1_1 = self.res1(h1, embed)  # 64
#         h2_1 = self.res2(h2, embed)
#         h3_1 = self.res3(h3, embed)
#
#         h4 = self.res4(torch.cat([h1_1, h2_1, h3_1], dim=1))  # 64
#         h5 = self.down1(h4, embed)  # 128
#         h6 = self.down2(h5, embed)  # 256
#         h = self.up1(h6, embed)  # 128
#
#         h += self.dense2(embed)
#         h = self.groupnorm2(h)
#         h = self.act(h)
#         h = self.up2(torch.cat([h, h5], dim=1))  # 64
#
#         h += self.dense3(embed)
#         h = self.groupnorm3(h)
#         h = self.act(h)
#         h = self.up3(torch.cat([h, h4], dim=1))  # 32
#
#         h += self.dense4(embed)
#         h = self.groupnorm4(h)
#         h = self.act(h)
#         h = self.final1(torch.cat([h, h1, h2, h3], dim=1))  # 32
#
#         h += self.dense5(embed)
#         h = self.groupnorm5(h)
#         h = self.act(h)
#         h = self.final2(h)  # 最后一个输出层我加了一个激活层控制
#         return h


class Multi_branch_Unet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=None, embed_dim=256, inter_dim=32):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(4, 32, kernel_size=3, stride=1)

        self.res1 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)
        self.res3 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)

        self.res4 = nn.Conv2d(channels[1] * 2, channels[1], 3, 1, padding=1)

        self.down1 = ResblockDownOne(channels[1], channels[2], embed_dim, 32)  # 三个分支合并，通道数量*3

        self.down2 = ResblockDownOne(channels[2], channels[3], embed_dim, 32)
        self.middle1 = ResblockOne(channels[3], channels[3], embed_dim, 32)
        self.middle2 = ResblockOne(channels[3], channels[3], embed_dim, 32)
        self.up1 = ResblockUpOne(channels[3], channels[2], embed_dim, 32)

        self.dense2 = Dense(embed_dim, channels[2])
        self.groupnorm2 = nn.GroupNorm(32, num_channels=channels[2])
        self.up2 = nn.ConvTranspose2d(in_channels=channels[2] + channels[2], out_channels=channels[1], kernel_size=3,
                                      stride=2, output_padding=1)

        self.dense3 = Dense(embed_dim, channels[1])
        self.groupnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.up3 = nn.ConvTranspose2d(in_channels=channels[1] + channels[1], out_channels=channels[0],
                                      kernel_size=3, stride=2, output_padding=1)

        self.dense4 = Dense(embed_dim, channels[0])
        self.groupnorm4 = nn.GroupNorm(4, num_channels=channels[0])
        self.final1 = nn.ConvTranspose2d(in_channels=channels[0] + channels[0] * 2,
                                         out_channels=channels[0], kernel_size=3, stride=1)

        self.dense5 = Dense(embed_dim, channels[0])
        self.groupnorm5 = nn.GroupNorm(4, num_channels=channels[0])
        self.final2 = nn.Conv2d(channels[0] + 8, channels[0], kernel_size=3, stride=1, padding=1)

        self.dense6 = Dense(embed_dim, channels[0])
        self.groupnorm6 = nn.GroupNorm(4, num_channels=channels[0])
        self.final3 = nn.Conv2d(channels[0], 4, kernel_size=3, stride=1, padding=1)

        self.act = SiLU()

    def forward(self, x_t, t_input, cond):  # 修改模型输入内容，修改为MS,PAN,NOISE三个部分，不能切片输入
        # Obtain the Gaussian random feature embedding for t
        t = t_input.view(-1, )
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理

        h1 = self.conv1(cond)  # 32
        h3 = self.conv3(x_t)

        h1_1 = self.res1(h1, embed)  # 64
        h3_1 = self.res3(h3, embed)

        h4 = self.res4(torch.cat([h1_1, h3_1], dim=1))  # 64

        h5 = self.down1(h4, embed)  # 128
        h6 = self.down2(h5, embed)  # 256

        h6 = self.middle1(self.middle2(h6, embed), embed)
        # 加入一个中间层结构
        h = self.up1(h6, embed)  # 128

        h += self.dense2(embed)
        h = self.groupnorm2(h)
        h = self.act(h)
        h = self.up2(torch.cat([h, h5], dim=1))  # 64

        h += self.dense3(embed)
        h = self.groupnorm3(h)
        h = self.act(h)
        h = self.up3(torch.cat([h, h4], dim=1))  # 32

        h += self.dense4(embed)
        h = self.groupnorm4(h)
        h = self.act(h)
        h = self.final1(torch.cat([h, h1, h3], dim=1))  # 32

        h += self.dense5(embed)
        h = self.groupnorm5(h)
        h = self.act(h)
        h = self.final2(torch.cat([h, cond, x_t], dim=1))  # 最后一个输出层我加了一个激活层控制

        # 加入一个输出层结构
        h += self.dense6(embed)
        h = self.groupnorm6(h)
        h = self.act(h)
        h = self.final3(h)
        return h
