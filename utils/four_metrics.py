import numpy as np
import os
import SimpleITK as sitk
from medpy import metric

def dice_coefficient(prediction, target, class_num = 14):
    dice_coefficient = []

    for i in range(class_num - 1):
        dice_cls = metric.binary.dc(prediction == (i + 1), target == (i + 1))
        dice_coefficient.append(dice_cls)

    return dice_coefficient

def jaccard_coefficient(prediction, target, slice,class_num = 14 ):

    jaccard_coefficient = []
    a = np.unique(target)
    for i in range(class_num - 1):
        try:
            dice_cls = metric.binary.jc(prediction == (i + 1), target == (i + 1))
            jaccard_coefficient.append(dice_cls)
        except ZeroDivisionError:
            pass

    return sum(jaccard_coefficient)/len(jaccard_coefficient)

def calculate_metrics(pred_folder, label_folder):
    pred_files = [os.path.join(pred_folder, file) for file in os.listdir(pred_folder)]
    label_files = [os.path.join(label_folder, file) for file in os.listdir(label_folder)]

    accuracy_lists = []
    class_accuracy = [0.0] * 13
    num_samples = 14   
    jaccard_scores = []
    num_classes = 14

    dice_lists = {}
    dice_lists['dice1'] = []
    dice_lists['dice2'] = []
    dice_lists['dice3'] = []
    dice_lists['dice4'] = []
    dice_lists['dice5'] = []
    dice_lists['dice6'] = []
    dice_lists['dice7'] = []
    dice_lists['dice8'] = []
    dice_lists['dice9'] = []
    dice_lists['dice10'] = []
    dice_lists['dice11'] = []
    dice_lists['dice12'] = []
    dice_lists['dice13'] = []

    for pred_path, label_path in zip(pred_files, label_files):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
        dice_scores = dice_coefficient(pred, label, num_classes)
        accuracy_lists.append(dice_scores)
        for class_id, dice_score in enumerate(dice_scores, start=1):
            print(f"Class {class_id} Dice Score: {dice_score*100:.2f}")
            dice_lists['dice' + str(class_id)].append(round(dice_score*100, 2))


        jaccard_scores.append(jaccard_coefficient(pred, label, slice))

    print(dice_lists['dice1'])
    print(dice_lists['dice2'])
    print(dice_lists['dice3'])
    print(dice_lists['dice4'])
    print(dice_lists['dice5'])
    print(dice_lists['dice6'])
    print(dice_lists['dice7'])
    print(dice_lists['dice8'])
    print(dice_lists['dice9'])
    print(dice_lists['dice10'])
    print(dice_lists['dice11'])
    print(dice_lists['dice12'])
    print(dice_lists['dice13'])

    for accuracy_list in accuracy_lists:
        for class_id in range(num_classes-1):
            class_accuracy[class_id] += accuracy_list[class_id]

    # 计算每个类别的平均精度
    average_accuracy = [acc / num_samples for acc in class_accuracy]

    average_accuracy[8] = average_accuracy[8] * 14 / 10  

    avg_jaccard = np.mean(jaccard_scores)


    # 打印每个类别的平均精度
    for class_id, avg_acc in enumerate(average_accuracy, start=1):
        print(f"Class {class_id} Average Accuracy: {avg_acc*100:.2f}")

    print( 'Average Accuracy', np.mean(average_accuracy))
    print('Average jaccard', avg_jaccard)

    return round(np.mean(average_accuracy)*100, 2), round(average_accuracy[0]*100, 2),round(average_accuracy[1]*100, 2),round(average_accuracy[2]*100, 2),round(average_accuracy[3]*100, 2),\
           round(average_accuracy[4]*100, 2),round(average_accuracy[5]*100, 2),round(average_accuracy[6]*100, 2),round(average_accuracy[7]*100, 2),round(average_accuracy[8]*100, 2),\
           round(average_accuracy[9]*100, 2), round(average_accuracy[10]*100, 2),round(average_accuracy[11]*100, 2),round(average_accuracy[12]*100, 2),round(avg_jaccard*100, 2)

