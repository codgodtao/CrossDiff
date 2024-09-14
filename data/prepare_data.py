from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py

# 读取文件夹下的所有mat，下采样上采样后保存为h5格式，方便后续的处理与计算！
# 64*64*4,256*256*1 无参考图像
# root = "/data/qlt/WV2_allmat"
# test_data_name = root + "/train"
# ms_save_path = test_data_name + '/ms'
# pan_save_path = test_data_name + '/pan'
#
# imgs_ms = os.listdir(ms_save_path)  # the name list of the images
# imgs_ms.sort(key=lambda x: int(x.split('.')[0]))
# imgs_ms = [test_data_name + '/ms/' + img for img in imgs_ms]  # the whole path of images list
# imgs_pan = os.listdir(pan_save_path)  # the name list of the images
# imgs_pan.sort(key=lambda x: int(x.split('.')[0]))
# imgs_pan = [test_data_name + '/pan/' + img for img in imgs_pan]  # the whole path of images list
# print('imgs_ms : ', imgs_ms)
# print('imgs_pan : ', imgs_pan)
# ratio = 4
# pan_list = torch.zeros(size=(1, 1, 256, 256))
# ms_list = torch.zeros(size=(1, 8, 64, 64))
# lms_list = torch.zeros(size=(1, 8, 256, 256))
# for item in range(len(imgs_ms)):
#     img_path_ms = imgs_ms[item]
#     img_path_pan = imgs_pan[item]
#     img_ms = scio.loadmat(img_path_ms)['ms']
#     img_pan = scio.loadmat(img_path_pan)['pan']
#
#     print(img_ms.shape, img_pan.shape)
#     # ref_np = img_ms
#     # gai
#     ms_size = img_ms.shape[1]
#     img_pan = img_pan.reshape(ms_size * 4, ms_size * 4, 1)
#     '''To shrink an image, it will generally look best with #INTER_AREA interpolation'''
#     # pan_np = cv2.resize(img_pan, (ms_size, ms_size), interpolation=cv2.INTER_AREA).reshape(ms_size, ms_size, 1)
#     # ms_np = cv2.resize(cv2.GaussianBlur(img_ms, (5, 5), 2),
#     #                    (ms_size // ratio, ms_size // ratio), interpolation=cv2.INTER_AREA)  # gai
#
#     '''data format change'''
#     # ref_torch_hwc = torch.from_numpy(ref_np).type(torch.FloatTensor)
#     pan_torch_hwc = torch.from_numpy(img_pan).type(torch.FloatTensor)
#     ms_torch_hwc = torch.from_numpy(img_ms).type(torch.FloatTensor)
#
#     '''channels change position'''
#     pan = pan_torch_hwc.permute(2, 0, 1)  # 1*256*256
#     ms = ms_torch_hwc.permute(2, 0, 1)  # 4*64*64
#
#     pan = pan.unsqueeze(0)
#     ms = ms.unsqueeze(0)
#     print(pan.shape, ms.shape)
#     pan_list = torch.concat((pan_list, pan), 0)
#     ms_list = torch.concat((ms_list, ms), 0)
# # 删除第一个维度的数据,使用基础的切片操作
# pan_list = pan_list[1:, :, :, :]
# ms_list = ms_list[1:, :, :, :]
# lms_list = F.interpolate(ms_list, scale_factor=4, mode="bicubic")  # only 3d,4d,5d support，需要将前面的batchsize设置好
# # 在其中第一个维度将他们加起来
# print(pan_list.shape, ms_list.shape, lms_list.shape)
#
# file_name = "/data/qlt/h5/full/training_data/train_wv2_data.h5"
# f = h5py.File(file_name, 'w')
# f.create_dataset('ms', data=ms_list)
# f.create_dataset('pan', data=pan_list)
# f.create_dataset('lms', data=lms_list)
# f.close()

# # # reduced resolution pan256*256 ms64*64*4 lms256*256*4 gt 256*256*4
# # # full resolution pan1024*1024 ms256*256*4 lms1024*1024*4 gt不存在
#
# ref_list = torch.zeros(size=(1, 4, 256, 256))
# pan_list = torch.zeros(size=(1, 1, 1024, 1024))
# ms_list = torch.zeros(size=(1, 4, 256, 256))
# lms_list = torch.zeros(size=(1, 4, 1024, 1024))
# for item in range(len(imgs_ms)):
#     img_path_ms = imgs_ms[item]
#     img_path_pan = imgs_pan[item]
#     img_ms = scio.loadmat(img_path_ms)['ms']
#     img_pan = scio.loadmat(img_path_pan)['pan'].reshape(1024, 1024, 1)
#     # img_ms = img_ms / 2047  # normalization
#     # img_pan = img_pan / 2047  # normalization
#
#     # print(ms_size)
#     # ref_np = img_ms
#     # gai
#     # ms_size = img_ms.shape[1]
#     # '''To shrink an image, it will generally look best with #INTER_AREA interpolation'''
#     # pan_np = cv2.resize(img_pan, (ms_size, ms_size), interpolation=cv2.INTER_AREA).reshape(ms_size, ms_size, 1)
#     # ms_np = cv2.resize(cv2.GaussianBlur(img_ms, (5, 5), 2),
#     #                    (ms_size // ratio, ms_size // ratio), interpolation=cv2.INTER_AREA)  # gai
#
#     '''data format change'''
#     # ref_torch_hwc = torch.from_numpy(ref_np).type(torch.FloatTensor)
#     pan_torch_hwc = torch.from_numpy(img_pan).type(torch.FloatTensor)
#     ms_torch_hwc = torch.from_numpy(img_ms).type(torch.FloatTensor)
#
#     '''channels change position'''
#     # ref = ref_torch_hwc.permute(2, 0, 1)  # 4*64*64
#     pan = pan_torch_hwc.permute(2, 0, 1)  # 1*64*64
#     ms = ms_torch_hwc.permute(2, 0, 1)  # 4*16*16
#
#     # ref = ref.unsqueeze(0)
#     pan = pan.unsqueeze(0)
#     ms = ms.unsqueeze(0)
#     print(pan.shape, ms.shape)
#     # ref_list = torch.concat((ref_list, ref), 0)
#     pan_list = torch.concat((pan_list, pan), 0)
#     ms_list = torch.concat((ms_list, ms), 0)
# # 删除第一个维度的数据,使用基础的切片操作
# # ref_list = ref_list[1:, :, :, :]
# pan_list = pan_list[1:, :, :, :]
# ms_list = ms_list[1:, :, :, :]
# lms_list = F.interpolate(ms_list, scale_factor=4, mode="bicubic")  # only 3d,4d,5d support，需要将前面的batchsize设置好
# # 在其中第一个维度将他们加起来
# print(pan_list.shape, ms_list.shape, lms_list.shape)
#
# # 转换为h5格式
# import h5py
#
# file_name = "test_wv2_data_FR.h5"
# f = h5py.File(file_name, 'w')
# f.create_dataset('ms', data=ms_list)
# # f.create_dataset('gt', data=ref_list)
# f.create_dataset('pan', data=pan_list)
# f.create_dataset('lms', data=lms_list)
# f.close()

"""
    验证保存的格式
"""
root = "D:/遥感图像融合/val_wv2_data.h5"
data = h5py.File(root)
pan = data['pan']
ms = data['ms']
lms = data['lms']
print("read_file_validation",pan.shape, ms.shape, lms.shape)
# 训练数据集的格式
# (9000, 4, 64, 64) (9000, 4, 16, 16) (9000, 4, 64, 64) (9000, 1, 64, 64)
# 我们的训练数据集格式
# (6943, 4, 64, 64) (6943, 4, 16, 16) (6943, 4, 64, 64) (6943, 1, 64, 64)
# 验证数据集格式
# (1000, 4, 64, 64) (1000, 4, 16, 16) (1000, 4, 64, 64) (1000, 1, 64, 64)
# 我们的验证数据集格式
# (743, 4, 64, 64) (743, 4, 16, 16) (743, 4, 64, 64) (743, 1, 64, 64)
# 测试数据集格式
# (7, 4, 512, 512) (7, 4, 128, 128) (7, 4, 512, 512) (7, 1, 512, 512)
# 我们的reduced测试数据集格式
# (156, 4, 256, 256) (156, 4, 64, 64) (156, 4, 256, 256) (156, 1, 256, 256)
# full测试数据集格式
# (4,2048,2048) (4,512,512) (1,2048,2048)
# 我们的full测试数据集格式
# (156, 4, 256, 256) (156, 4, 1024, 1024) (156, 1, 1024, 1024)
