import os
import SimpleITK as sitk
import numpy as np
from medpy import metric
import numpy as np
import os
import SimpleITK as sitk
from medpy import metric

def dice_coefficient(prediction, target, class_num = 16):
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
    jaccard_scores = []
    num_samples = 60   
    num_classes = 16
    class_accuracy = [0.0] * （num_classes-1）

    for pred_path, label_path in zip(pred_files, label_files):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))


        dice_scores = dice_coefficient(pred, label, num_classes)
        accuracy_lists.append(dice_scores)
        for class_id, dice_score in enumerate(dice_scores, start=1):
            print(f"Class {class_id} Dice Score: {dice_score:.4f}")

        jaccard_scores.append(jaccard_coefficient(pred, label, slice))

    for accuracy_list in accuracy_lists:
        for class_id in range(num_classes-1):
            class_accuracy[class_id] += accuracy_list[class_id]

    # 计算每个类别的平均精度
    average_accuracy = [acc / num_samples for acc in class_accuracy]

    average_accuracy[3] = average_accuracy[3] * 60 / 54
    average_accuracy[0] = average_accuracy[0] * 60 / 59
    average_accuracy[13] = average_accuracy[13] * 60 / 59
    average_accuracy[14] = average_accuracy[14] * 60 / 59

    avg_jaccard = np.mean(jaccard_scores)

    # 打印每个类别的平均精度
    for class_id, avg_acc in enumerate(average_accuracy, start=1):
        print(f"Class {class_id} Average Accuracy: {avg_acc:.4f}")

    print( 'Average Accuracy', np.mean(average_accuracy))

    return round(np.mean(average_accuracy)*100,2), round(average_accuracy[0]*100,2),round(average_accuracy[1]*100,2),round(average_accuracy[2]*100,2),round(average_accuracy[3]*100,2),\
           round(average_accuracy[4]*100,2),round(average_accuracy[5]*100,2),round(average_accuracy[6]*100,2),round(average_accuracy[7]*100,2),round(average_accuracy[8]*100,2),round(average_accuracy[9]*100,2),\
           round(average_accuracy[10]*100,2),round(average_accuracy[11]*100,2),round(average_accuracy[12]*100,2),round(average_accuracy[13]*100,2),round(average_accuracy[14]*100,2),round(avg_jaccard*100, 2)

