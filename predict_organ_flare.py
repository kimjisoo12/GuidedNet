import os
import argparse

import math
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import logging
from utils.four_metrics import calculate_metrics

from networks.unet_3D_gmm import unet_gmm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--ratio', type=int, default='10', help='laebeled data to use')
FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def extract_categories(label_image):
    unique_classes = np.unique(label_image)
    return unique_classes.tolist()

def test_all_case(net, imdir, maskdir, jisoo, output2, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for pdx, fname in enumerate(sorted(getFiles(imdir))):
        # load files
        print(f"Processing {fname.replace('_0000.nii.gz', '')}")
        sitk_im = sitk.ReadImage(os.path.join(imdir, fname))  # img
        im_x_y = sitk.GetArrayFromImage(sitk_im)  # zyx

        sitk_mask = sitk.ReadImage(os.path.join(maskdir, fname))  # mask
        label = sitk.GetArrayFromImage(sitk_mask)

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net,  jisoo,  label, im_x_y, stride_xy, stride_z, patch_size,
                                                 num_classes=num_classes)  # zyx

        prediction = prediction.astype(np.uint8)

        categories = extract_categories(prediction)  
        print(categories)

        saveprediction = sitk.GetImageFromArray(prediction)
        saveprediction.SetSpacing(sitk_im.GetSpacing())
        saveprediction.SetOrigin(sitk_im.GetOrigin())
        saveprediction.SetDirection(sitk_im.GetDirection())


        sitk.WriteImage(saveprediction, output2 + fname.split('_')[0] + "_"
                        + fname.split('_')[1] + ".nii.gz")


def test_single_case(net,  jisoo, label,  image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_z) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_xy) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_z * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_xy * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(
                    np.float32)  # 添加一个维度 ex:(1,512,512)到(1,1,512,512)
                test_patch = torch.from_numpy(test_patch).cuda()  

                if jisoo == 1:
                    y1 = net(test_patch)
                    y = F.softmax(y1, dim=1)

                elif jisoo == 2:
                    y1_tanh, y1 = net(test_patch)
                    y = F.softmax(y1, dim=1)
                elif jisoo == 33:
                    y1_tanh, y1 = net(test_patch)
                    y = F.softmax(y1_tanh, dim=1)
                else:
                    y1 = net(test_patch)
                    y = F.softmax(y1['pred'], dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]  # (15,112,112,80)分别对应着15类（0，1划分的）三维数组
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def test_calculate_metric():

    imdir = "infer/test/"     # test data
    snapshot_path = "infer/flare/"  + str(FLAGS.ratio) + '/' # Checkpoints
    output2 = 'infer/predict/'     # prediction
    path1 = output2
    path2 = "infer/test_label/"   # test label
    # log
    logging.basicConfig(filename= str(FLAGS.ratio) + '_guidedNet.log',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    num_classes = 14


    folder_path = snapshot_path
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        net = unet_gmm(in_channels=1, n_classes=num_classes).cuda()
        jisoo = 3

        save_mode_path = os.path.join(snapshot_path, file_name)
        folder_path = snapshot_path + file_name
        file_list = os.listdir(folder_path)

        avg_dice_final = []
        avg_dice1 = []
        avg_dice2 = []
        avg_dice3 = []
        avg_dice4 = []
        avg_dice5 = []
        avg_dice6 = []
        avg_dice7 = []
        avg_dice8 = []
        avg_dice9 = []
        avg_dice10 = []
        avg_dice11 = []
        avg_dice12 = []
        avg_dice13 = []
        avg_jaccard_final = []


        for file_name in file_list:
            save_mode_path_new = os.path.join(save_mode_path, file_name)
            logging.info("### Now model is {}".format(save_mode_path_new))
            net.load_state_dict(torch.load(save_mode_path_new))
            print("### init weight from {}".format(save_mode_path_new))
            net.eval()

            test_all_case(net, imdir, path2, jisoo,  output2, num_classes=num_classes,
                          patch_size=(64, 160, 160), stride_xy=80, stride_z=32, save_result=True)

            avg_dice, organ1, organ2, organ3, organ4, organ5, organ6, organ7, organ8, organ9, organ10, \
            organ11, organ12, organ13, avg_jaccard = calculate_metrics(path1, path2)

            #  single checkpoints
            logging.info('avg_dice:{}'.format(avg_dice))
            logging.info('organ1:{}'.format(organ1))
            logging.info('organ2:{}'.format(organ2))
            logging.info('organ3:{}'.format(organ3))
            logging.info('organ4:{}'.format(organ4))
            logging.info('organ5:{}'.format(organ5))
            logging.info('organ6:{}'.format(organ6))
            logging.info('organ7:{}'.format(organ7))
            logging.info('organ8:{}'.format(organ8))
            logging.info('organ9:{}'.format(organ9))
            logging.info('organ10:{}'.format(organ10))
            logging.info('organ11:{}'.format(organ11))
            logging.info('organ12:{}'.format(organ12))
            logging.info('organ13:{}'.format(organ13))
            logging.info('avg_jaccard:{}'.format(avg_jaccard))
            logging.info('$' + str(organ1) +'$&$' + str(organ3) +'$&$'+ str(organ11)+'$&$'+ str(organ13)+'$&$'+ str(organ2)+'$&$'+ str(organ5)+'$&$'+ str(organ4)+'$&$'+ str(organ6)
                         +'$&$'+ str(organ12)+'$&$'+ str(organ9)+'$&$'+ str(organ10)+'$&$'+ str(organ7)+'$&$'+ str(organ8)+'$&$'+ str(avg_dice)+'$&$'+ str(avg_jaccard))


            avg_dice_final.append(avg_dice)
            avg_dice1.append(organ1)
            avg_dice2.append(organ2)
            avg_dice3.append(organ3)
            avg_dice4.append(organ4)
            avg_dice5.append(organ5)
            avg_dice6.append(organ6)
            avg_dice7.append(organ7)
            avg_dice8.append(organ8)
            avg_dice9.append(organ9)
            avg_dice10.append(organ10)
            avg_dice11.append(organ11)
            avg_dice12.append(organ12)
            avg_dice13.append(organ13)
            avg_jaccard_final.append(avg_jaccard)

        # all checkpoints
        logging.info('avg_dice:{}'.format(np.mean(avg_dice_final)))
        logging.info('std:{}'.format(np.std(avg_dice_final)))

        logging.info('Organ 1 mean:{}'.format(np.mean(avg_dice1)))
        logging.info(' Organ 1 std:{}'.format(np.std(avg_dice1)))

        logging.info('Organ 2 mean:{}'.format(np.mean(avg_dice2)))
        logging.info(' Organ 2 std:{}'.format(np.std(avg_dice2)))

        logging.info('Organ 3 mean:{}'.format(np.mean(avg_dice3)))
        logging.info(' Organ 3 std:{}'.format(np.std(avg_dice3)))

        logging.info('Organ 4 mean:{}'.format(np.mean(avg_dice4)))
        logging.info(' Organ 4 std:{}'.format(np.std(avg_dice4)))

        logging.info('Organ 5 mean:{}'.format(np.mean(avg_dice5)))
        logging.info(' Organ 5 std:{}'.format(np.std(avg_dice5)))

        logging.info('Organ 6 mean:{}'.format(np.mean(avg_dice6)))
        logging.info(' Organ 6 std:{}'.format(np.std(avg_dice6)))

        logging.info('Organ 7 mean:{}'.format(np.mean(avg_dice7)))
        logging.info(' Organ 7 std:{}'.format(np.std(avg_dice7)))

        logging.info('Organ 8 mean:{}'.format(np.mean(avg_dice8)))
        logging.info(' Organ 8 std:{}'.format(np.std(avg_dice8)))

        logging.info('Organ 9 mean:{}'.format(np.mean(avg_dice9)))
        logging.info(' Organ 9 std:{}'.format(np.std(avg_dice9)))

        logging.info('Organ 10 mean:{}'.format(np.mean(avg_dice10)))
        logging.info(' Organ 10 std:{}'.format(np.std(avg_dice10)))

        logging.info('Organ 11 mean:{}'.format(np.mean(avg_dice11)))
        logging.info(' Organ 11 std:{}'.format(np.std(avg_dice11)))

        logging.info('Organ 12 mean:{}'.format(np.mean(avg_dice12)))
        logging.info(' Organ 12 std:{}'.format(np.std(avg_dice12)))

        logging.info('Organ 13 mean:{}'.format(np.mean(avg_dice13)))
        logging.info(' Organ 13 std:{}'.format(np.std(avg_dice13)))

        logging.info('avg_jaccard:{}'.format(np.mean(avg_jaccard_final)))
        logging.info('std:{}'.format(np.std(avg_jaccard_final)))


        std_dice_final = round(np.std(avg_dice_final) , 2)
        std_dice1 = round(np.std(avg_dice1) , 2)
        std_dice2 = round(np.std(avg_dice1), 2)
        std_dice3 = round(np.std(avg_dice1), 2)
        std_dice4 = round(np.std(avg_dice1), 2)
        std_dice5 = round(np.std(avg_dice1), 2)
        std_dice6 = round(np.std(avg_dice1), 2)
        std_dice7 = round(np.std(avg_dice1), 2)
        std_dice8 = round(np.std(avg_dice1), 2)
        std_dice9 = round(np.std(avg_dice9), 2)
        std_dice10 = round(np.std(avg_dice10), 2)
        std_dice11 = round(np.std(avg_dice11), 2)
        std_dice12 = round(np.std(avg_dice12), 2)
        std_dice13 = round(np.std(avg_dice13), 2)
        std_jaccard_final = round(np.std(avg_jaccard_final), 2)

        avg_dice = round(np.mean(avg_dice_final) , 2)
        organ1 = round(np.mean(avg_dice1), 2)
        organ2 = round(np.mean(avg_dice2), 2)
        organ3 = round(np.mean(avg_dice3), 2)
        organ4 = round(np.mean(avg_dice4), 2)
        organ5 = round(np.mean(avg_dice5), 2)
        organ6 = round(np.mean(avg_dice6), 2)
        organ7 = round(np.mean(avg_dice7), 2)
        organ8 = round(np.mean(avg_dice8), 2)
        organ9 = round(np.mean(avg_dice9), 2)
        organ10 = round(np.mean(avg_dice10), 2)
        organ11 = round(np.mean(avg_dice11), 2)
        organ12= round(np.mean(avg_dice12), 2)
        organ13 = round(np.mean(avg_dice13), 2)
        avg_jaccard = round(np.mean(avg_jaccard_final), 2)

        logging.info(
            '$' + str(organ1) + '$&$' + str(organ3) + '$&$' + str(organ11) + '$&$' + str(organ13) + '$&$' + str(
                organ2) + '$&$' + str(organ5) + '$&$' + str(organ4) + '$&$' + str(organ6)
            + '$&$' + str(organ12) + '$&$' + str(organ9) + '$&$' + str(organ10) + '$&$' + str(organ7) + '$&$' + str(
                organ8) + '$&$' + str(avg_dice) + '\pm'+ str(std_dice_final) + '$&$' + str(avg_jaccard) + '\pm'+ str(std_jaccard_final) +'$')




if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)