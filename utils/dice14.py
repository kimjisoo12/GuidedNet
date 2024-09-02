import numpy as np
import nibabel as nib
import os
from medpy import metric
def cal_subject_level_dice(prediction, target, class_num=15):# class_num是你分割的目标的类别个数
    '''
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param class_num: total number of categories
    :return:
    '''

    dscs = []
    a = np.unique(target)

    if 9 not in a :
        print('标签里没有9')
        for i in range(class_num):
            if i == 9:
                continue
            else:
                dice_cls = metric.binary.dc(prediction == (i), target == (i))
                dscs.append(dice_cls)
    else:
        print('标签里有9')
        for i in range(class_num):
            dice_cls = metric.binary.dc(prediction == (i), target == (i))
            dscs.append(dice_cls)
    subject_level_dice = sum(dscs[1:]) / (len(dscs) - 1)

    num_1 = dscs[1]
    num_2 = dscs[2]
    num_3 = dscs[3]
    num_4 = dscs[4]
    num_5 = dscs[5]
    num_6 = dscs[6]
    num_7 = dscs[7]
    num_8 = dscs[8]
    num_9 = dscs[9]
    num_10 = dscs[10]
    num_11 = dscs[11]
    num_12 = dscs[12]
    num_13 = dscs[13]

    return subject_level_dice, num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8,num_9,num_10, num_11, num_12, num_13

def evaluate_demo(prediction_nii_files, target_nii_files):
    '''
    This is a demo for calculating the mean dice of all subjects.
    :param prediction_nii_files: a list which contains the .nii file paths of predicted segmentation
    :param target_nii_files: a list which contains the .nii file paths of ground truth mask
    :return:
    '''
    dscs = []
    dscs1 = []
    dscs2 = []
    dscs3 = []
    dscs4 = []
    dscs5 = []
    dscs6 = []
    dscs7 = []
    dscs8 = []
    dscs9 = []
    dscs10 = []
    dscs11 = []
    dscs12 = []
    dscs13 = []


    for prediction_nii_file, target_nii_file in zip(prediction_nii_files, target_nii_files):
        prediction_nii = nib.load(prediction_nii_file)
        prediction = prediction_nii.get_data()
        target_nii = nib.load(target_nii_file)
        target = target_nii.get_data()
        dsc, dsc1, dsc2, dsc3, dsc4, dsc5, dsc6, dsc7, dsc8, dsc9, dsc10, dsc11, dsc12, dsc13 = cal_subject_level_dice(prediction, target, class_num=14)
        print(prediction_nii_file.split('/')[-1] + "---的dice值为---" + str(format(dsc,"0.4f")) )# 保留四位小数
        dscs.append(dsc)


        print(prediction_nii_file.split('/')[-1] + "---1的dice值为---" + str(format(dsc1, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---2的dice值为---" + str(format(dsc2, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---3的dice值为---" + str(format(dsc3, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---4的dice值为---" + str(format(dsc4, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---5的dice值为---" + str(format(dsc5, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---6的dice值为---" + str(format(dsc6, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---7的dice值为---" + str(format(dsc7, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---8的dice值为---" + str(format(dsc8, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---9的dice值为---" + str(format(dsc9, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---10的dice值为---" + str(format(dsc10, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---11的dice值为---" + str(format(dsc11, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---12的dice值为---" + str(format(dsc12, "0.4f")))  # 保留四位小数
        print(prediction_nii_file.split('/')[-1] + "---13的dice值为---" + str(format(dsc13, "0.4f")))  # 保留四位小数

        print('------------------------------------------------')
        dscs1.append(dsc1)
        dscs2.append(dsc2)
        dscs3.append(dsc3)
        dscs4.append(dsc4)
        dscs5.append(dsc5)
        dscs6.append(dsc6)
        dscs7.append(dsc7)
        dscs8.append(dsc8)
        dscs9.append(dsc9)
        dscs10.append(dsc10)
        dscs11.append(dsc11)
        dscs12.append(dsc12)
        dscs13.append(dsc13)


    #return np.mean(dscs)
    return format(dsc,"0.4f"), format(dsc1,"0.4f"), format(dsc2,"0.4f"), format(dsc3,"0.4f"), format(dsc4,"0.4f"), format(dsc5,"0.4f"),format(dsc6,"0.4f"), format(dsc7,"0.4f"), format(dsc8,"0.4f"), format(dsc9,"0.4f"), format(dsc10,"0.4f"), format(dsc11,"0.4f"), format(dsc12,"0.4f"),format(dsc13,"0.4f")

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls
def dice(imdir,outdir):
    dice_list = []
    dice_list1 = []
    dice_list2 = []
    dice_list3 = []
    dice_list4 = []
    dice_list5 = []
    dice_list6 = []
    dice_list7 = []
    dice_list8 = []
    dice_list9 = []
    dice_list10 = []
    dice_list11 = []
    dice_list12 = []
    dice_list13 = []

    for pdx, fname in enumerate(sorted(getFiles(imdir))):

        dice, dice1, dice2, dice3, dice4, dice5, dice6, dice7, dice8, \
        dice9, dice10, dice11, dice12, dice13, = evaluate_demo([outdir  + fname ],[imdir + fname ])

        dice_list.append(dice)
        dice_list1.append(dice1)
        dice_list2.append(dice2)
        dice_list3.append(dice3)
        dice_list4.append(dice4)
        dice_list5.append(dice5)
        dice_list6.append(dice6)
        dice_list7.append(dice7)
        dice_list8.append(dice8)
        dice_list9.append(dice9)
        dice_list10.append(dice10)
        dice_list11.append(dice11)
        dice_list12.append(dice12)
        dice_list13.append(dice13)

    sum = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0
    sum8 = 0
    sum9 = 0
    sum10 = 0
    sum11 = 0
    sum12 = 0
    sum13 = 0

    for item in dice_list:
        sum += float(item)
    mean_dice = sum/len(dice_list)
    print("mean——dice值为-----" + str(format(mean_dice,'0.4f')))

    for item in dice_list1:
        sum1 += float(item)
    mean_dice1 = sum1/len(dice_list1)
    print("mean——dice1值为-----" + str(format(mean_dice1,'0.4f')))

    for item in dice_list2:
        sum2 += float(item)
    mean_dice2 = sum2/len(dice_list2)
    print("mean——dice2值为-----" + str(format(mean_dice2,'0.4f')))

    for item in dice_list3:
        sum3 += float(item)
    mean_dice3 = sum3/len(dice_list3)
    print("mean——dice3值为-----" + str(format(mean_dice3,'0.4f')))

    for item in dice_list4:
        sum4 += float(item)
    mean_dice4 = sum4/len(dice_list4)
    print("mean——dice4值为-----" + str(format(mean_dice4,'0.4f')))

    for item in dice_list5:
        sum5 += float(item)
    mean_dice5 = sum5/len(dice_list5)
    print("mean——dice5值为-----" + str(format(mean_dice5,'0.4f')))

    for item in dice_list6:
        sum6 += float(item)
    mean_dice6 = sum6/len(dice_list6)
    print("mean——dice6值为-----" + str(format(mean_dice6,'0.4f')))

    for item in dice_list7:
        sum7 += float(item)
    mean_dice7 = sum7/len(dice_list7)
    print("mean——dice7值为-----" + str(format(mean_dice7,'0.4f')))

    for item in dice_list8:
        sum8 += float(item)
    mean_dice8 = sum8/len(dice_list8)
    print("mean——dice8值为-----" + str(format(mean_dice8,'0.4f')))

    for item in dice_list9:
        sum9 += float(item)
    mean_dice9 = sum9/len(dice_list9)
    print("mean——dice9值为-----" + str(format(mean_dice9,'0.4f')))

    for item in dice_list10:
        sum10 += float(item)
    mean_dice10 = sum10/len(dice_list10)
    print("mean——dice10值为-----" + str(format(mean_dice10,'0.4f')))

    for item in dice_list11:
        sum11 += float(item)
    mean_dice11 = sum11/len(dice_list11)
    print("mean——dice11值为-----" + str(format(mean_dice11,'0.4f')))

    for item in dice_list12:
        sum12 += float(item)
    mean_dice12 = sum12/len(dice_list12)
    print("mean——dice12值为-----" + str(format(mean_dice12,'0.4f')))

    for item in dice_list13:
        sum13 += float(item)
    mean_dice13 = sum13/len(dice_list13)
    print("mean——dice13值为-----" + str(format(mean_dice13,'0.4f')))

