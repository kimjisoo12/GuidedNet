import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from util.utils import *



def build_cur_cls_label(mask, nclass):
    """some point annotations are cropped out, thus the prototypes are partial"""
    b = mask.size()[0]
    mask_one_hot = one_hot(mask, nclass)
    a = mask_one_hot.view(b, nclass, -1)
    bb = a.max(-1)
    cur_cls_label = mask_one_hot.view(b, nclass, -1).max(-1)[0]
    return cur_cls_label.view(b, nclass, 1, 1, 1)


def clean_mask(mask, cls_label, softmax=True):
    if softmax:
        mask = F.softmax(mask, dim=1)
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1, 1)

def clean_mask_predict(mask,  softmax=True):
    if softmax:
        mask = F.softmax(mask, dim=1)
    """Remove any masks of labels that are not present"""
    return mask


def get_cls_loss(predict, cls_label, mask):
    """cls_label: (b, k)"""
    """ predict: (b, k, h, w)"""
    """ mask: (b, h, w) """
    b, k, d, h, w = predict.size()
    predict = torch.softmax(predict, dim=1).view(b, k, -1)
    mask = mask.view(b, -1)

    # if a patch does not contain label k,
    # then none of the pixels in this patch can be assigned to label k
    loss = - (1 - cls_label.view(b, k, 1)) * torch.log(1 - predict + 1e-6)
    loss = torch.sum(loss, dim=1)
    loss = loss[mask != 255].mean()
    return loss


def one_hot(label, nclass):
    b, d, h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(b, 1, d*h*w)

    mask = torch.zeros(b, nclass+1, d*h*w).to(label.device)
    mask = mask.scatter_(1, label_cp.long(), 1).view(b, nclass+1, d, h, w).float()
    return mask[:, :-1, :, :, :]

def one_hot_3d(label, nclass):
    d, h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(1, d*h*w)

    mask = torch.zeros(nclass+1, d*h*w).to(label.device)
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, d, h, w).float()
    return mask[:-1, :, :, :]

def cal_protypes(feat, mask, nclass):

    b, c, d, h, w = feat.size()
    prototypes = torch.zeros((b, nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)
    for i in range(b): # Read every image in each batch
        cur_mask = mask[i]  # The corresponding mask of each sheet is called cur_mask
        cur_mask_onehot = one_hot_3d(cur_mask, nclass) # Convert the mask to 0/1

        cur_feat = feat[i] # Each corresponding feature map is called a cur_feat
        cur_prototype = torch.zeros((nclass, c),
                           dtype=feat.dtype,
                           device=feat.device) # The feature center of all the categories in this diagram

        # Single-out distinct elements in the cur_mask indicate that there are several labels in this diagram
        cur_set = list(torch.unique(cur_mask))

        for cls in cur_set:
            m = cur_mask_onehot[cls].view(1, d, h, w) # This image contains all the locations in this category only
            sum = m.sum() # Count the total number of images in this category
            m = m.expand(c, d, h, w).view(c, -1)
            cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1)/(sum + 1e-6) # Calculate the feature center for the category
            cur_prototype[cls, :] = cls_feat

        #cur_prototype The feature centers of all categories in this diagram
        #prototypes The feature center of all categories for all images in a batch
        prototypes[i] += cur_prototype   

    cur_cls_label = build_cur_cls_label(mask, nclass).view(b, nclass, 1)
    mean_vecs = (prototypes.sum(0)*cur_cls_label.sum(0))/(cur_cls_label.sum(0)+1e-6)

    loss = proto_loss(prototypes, mean_vecs, cur_cls_label)

    return prototypes.view(b, nclass, c), loss


def proto_loss(prototypes, vecs, cur_cls_label):
    b, nclass, c = prototypes.size()

    # abs = torch.abs(prototypes - vecs).mean(2)
    # positive = torch.exp(-(abs * abs))
    # positive = (positive*cur_cls_label.view(b, nclass)).sum()/(cur_cls_label.sum()+1e-6)
    # positive_loss = 1 - positive

    vecs = vecs.view(nclass, c)
    total_cls_label = (cur_cls_label.sum(0) > 0).long()
    negative = torch.zeros(1,
                           dtype=prototypes.dtype,
                           device=prototypes.device)

    num = 0
    for i in range(nclass):
        if total_cls_label[i] == 1:
            for j in range(i+1, nclass):
                if total_cls_label[j] == 1:
                    if i != j:
                        num += 1
                        x, y = vecs[i].view(1, c), vecs[j].view(1, c)
                        abs = torch.abs(x - y).mean(1)
                        negative += torch.exp(-(abs * abs))
                        # print(negative)

    negative = negative/(num+1e-6)
    negative_loss = negative

    return negative_loss


def GMM(feat, vecs, pred, true_mask, cls_label):
    b, k, od, oh, ow = pred.size()

    preserve = (true_mask < 255).long().view(b, 1, od, oh, ow)
    _, _, d, h, w = pred.size()

    vecs = vecs.view(b, k, -1, 1, 1, 1)
    feat = feat.view(b, 1, -1, d, h, w)

    with torch.cuda.device(1):
        feat = feat.cuda()
        vecs = vecs.cuda()
        cls_label = cls_label.cuda()
        preserve = preserve.cuda()

    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs).mean(2)
    abs = abs * cls_label.view(b, k, 1, 1, 1) * preserve.view(b, 1, d, h, w)
    abs = abs.view(b, k, d*h*w)

    # """ calculate std """
    # pred = pred * preserve
    # num = pred.view(b, k, -1).sum(-1)
    # std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    # std = ((abs ** 2).sum(-1)/(preserve.view(b, 1, -1).sum(-1)) + 1e-6) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    abs = abs.view(b, k, d, h, w)
    res = torch.exp(-(abs * abs))
    res = res * cls_label.view(b, k, 1, 1, 1)
    
    res = res.cuda(0) 
    return res

def GMM_predict(feat, vecs, pred):
    b, k, od, oh, ow = pred.size()

    # preserve = (true_mask < 255).long().view(b, 1, od, oh, ow)
    _, _, d, h, w = pred.size()

    vecs = vecs.view(b, k, -1, 1, 1, 1)
    feat = feat.view(b, 1, -1, d, h, w)

    with torch.cuda.device(1):
        feat = feat.cuda()
        vecs = vecs.cuda()

    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs).mean(2)
    # abs = abs * cls_label.view(b, k, 1, 1, 1) * preserve.view(b, 1, d, h, w)
    abs = abs.view(b, k, d*h*w)

    abs = abs.view(b, k, d, h, w)
    res = torch.exp(-(abs * abs))

    res = res.cuda(0) 
    return res

def cal_gmm_loss(pred, res, cls_label, true_mask):
    n, k, d, h, w = pred.size()

    with torch.cuda.device(1):
        res = res.cuda()
        pred = pred.cuda()
        cls_label = cls_label.cuda()
        true_mask = true_mask.cuda()
  
    loss1 = - res * torch.log(pred + 1e-6) - (1 - res) * torch.log(1 - pred + 1e-6)
    loss1 = loss1/2
    loss1 = (loss1*cls_label).sum(1)/(cls_label.sum(1)+1e-6)
    loss1 = loss1.mean()

    true_mask_one_hot = one_hot(true_mask, k)
    true_mask_one_hot = true_mask_one_hot.half()
    
    loss2 = - true_mask_one_hot * torch.log(res + 1e-6) \
            - (1 - true_mask_one_hot) * torch.log(1 - res + 1e-6)
    loss2 = loss2/2
    loss2 = (loss2 * cls_label).sum(1) / (cls_label.sum(1) + 1e-6)
    loss2 = loss2[true_mask < k].mean()

    loss1 = loss1.cuda(0) 
    loss2 = loss2.cuda(0) 

    return loss1+loss2


def gmm_loss(pred, res, cls_label):

    res = res.half()
    pred = pred.half()

    with torch.cuda.device(1):
        res = res.cuda()
        pred = pred.cuda()
        cls_label = cls_label.cuda()

    loss1 = - res * torch.log(pred + 1e-6) - (1 - res) * torch.log(1 - pred + 1e-6)
    loss1 = loss1/2
    loss1 = (loss1*cls_label).sum(1)/(cls_label.sum(1)+1e-6)
    loss1 = loss1.mean()
 
    loss1 = loss1.cuda(0) 
    return loss1


