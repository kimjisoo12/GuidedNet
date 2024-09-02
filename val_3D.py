import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm

import threading
import random


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_z) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_xy) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_z*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_xy * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1['pred'], dim=1)
                    # y = torch.softmax(y1[0], dim=1)  magicnet用这行代码
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)






# 定义一个MyThread.py线程类
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

def test_single_case_batch(namesi, net, stride_xy, stride_z, patch_size, num_classes):
    total_metric = np.zeros((num_classes - 1, 2))

    for image_path in namesi:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)

    return total_metric / len(namesi)


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}/2022.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]

 # 使用random.shuffle()来随机打乱原始列表
    random.shuffle(image_list)

#将列表分成两组，每组包含15个元素
    group1 = image_list[:2]
    group2 = image_list[2:4]
    group3 = image_list[4:6]
    group4 = image_list[6:8]
    group5 = image_list[8:10]
    group6 = image_list[10:12]
    group7 = image_list[12:]

    more_th1 = MyThread(test_single_case_batch,(group1, net, stride_xy, stride_z, patch_size, num_classes))
    more_th2 = MyThread(test_single_case_batch, (group2, net, stride_xy, stride_z, patch_size, num_classes))
    more_th3 = MyThread(test_single_case_batch, (group3, net, stride_xy, stride_z, patch_size, num_classes))
    more_th4 = MyThread(test_single_case_batch, (group4, net, stride_xy, stride_z, patch_size, num_classes))
    more_th5 = MyThread(test_single_case_batch, (group5, net, stride_xy, stride_z, patch_size, num_classes))
    more_th6 = MyThread(test_single_case_batch, (group6, net, stride_xy, stride_z, patch_size, num_classes))
    more_th7 = MyThread(test_single_case_batch, (group7, net, stride_xy, stride_z, patch_size, num_classes))

    more_th1.start()
    more_th2.start()
    more_th3.start()
    more_th4.start()
    more_th5.start()
    more_th6.start()
    more_th7.start()


    more_th1.join()
    more_th2.join()
    more_th3.join()
    more_th4.join()
    more_th5.join()
    more_th6.join()
    more_th7.join()

    total_metric = more_th1.get_result() + more_th2.get_result()+ more_th3.get_result()\
                   + more_th4.get_result()+ more_th5.get_result()+ more_th6.get_result()+ more_th7.get_result()


    return total_metric / (len(image_list)/2)