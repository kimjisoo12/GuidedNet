import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.brats2019 import (BraTS2019, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler)
from net_factory_3d import net_factory_3d
from utils import losses, ramps, tools
from val_3D import test_all_case
import h5py
from utils.loss import RobustCrossEntropyLoss

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../dataa/2022_flare_10', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='test', help='experiment_name')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--patch_size', type=list,  default=[64, 128, 128], help='patch size of network input')
parser.add_argument('--labeled_num', type=int, default=42, help='labeled data')
parser.add_argument('--data_num', type=int, default=84, help='all data')
parser.add_argument('--model', type=str, default='guidedNet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.1, help='segmentation network learning rate')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--beta', type=float,  default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5, help='balance factor to control supervised and consistency loss')
parser.add_argument('--lanmuda', type=float, default=1, help='consistency')
args = parser.parse_args()


def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight

class DistDW:
    def __init__(self, num_cls, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P = (num_each_class.max()+1e-8) / (num_each_class+1e-8)
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, trainloader):
        num_each_class = np.zeros(self.num_cls)
        ids_list = trainloader.dataset.image_list[:42]
        for data_id in ids_list:
            h5f = h5py.File(args.root_path + "/{}/2022.h5".format(data_id), 'r')
            label = h5f['label'][:]
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label, label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy = pseudo_label.data.cpu().numpy()

        gt_numpy = label.data.cpu().numpy()
        label_numpy = np.squeeze(label_numpy, axis=1)

        mask = (label_numpy == gt_numpy)
        label_numpy = np.where(mask, label_numpy, 0)

        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp

        cur_weights = self._cal_weights(num_each_class) * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train(args, snapshot_path,):

    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 14
    best_performance1_test = 0.0
    best_performance2_test = 0.0

    net1 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    net2 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
 
    model1.train()
    model2.train()
    
    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.data_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    iter_num = 0

    from torch.nn import MSELoss
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    distdw = DistDW(num_classes, momentum=0.99) 
    weight_A = distdw.init_weights(trainloader)
    weight_B = distdw.init_weights(trainloader)
    
    start_time = time.time()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            out1 = model1(volume_batch)
            outputs1, rep1 = out1["pred"], out1["rep"]
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            out2 = model2(volume_batch)
            outputs2, rep2 = out2["pred"], out2["rep"]
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            
            # supversied loss
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))    

            # train Gaussian1
            feat1 = rep1[:args.labeled_bs] # fearure
            mask1 = label_batch[:args.labeled_bs] # mask
            cls_label1 = torch.stack([torch.arange(num_classes)] * args.labeled_bs).cuda() # class label
            cur_cls_label1 = tools.build_cur_cls_label(mask1, num_classes)
            pred_cl1 = tools.clean_mask(outputs1[:args.labeled_bs], cls_label1, True)
            vecs1, proto_loss1, = tools.cal_protypes(feat1, mask1, num_classes) 
            res1 = tools.GMM(feat1, vecs1, pred_cl1, mask1, cur_cls_label1)
            gmm_loss1 = tools.cal_gmm_loss(outputs_soft1[:args.labeled_bs], res1, cur_cls_label1, mask1) + proto_loss1           

            # Gaussian1 predict
            feat1_u = rep1[args.labeled_bs:] # fearure
            pred_cl1_u = tools.clean_mask_predict(outputs1[args.labeled_bs:], True)
            res1_u = tools.GMM_predict(feat1_u, vecs1, pred_cl1_u)
            gmm_loss1_u = tools.gmm_loss(outputs_soft1[args.labeled_bs:], res1_u, cur_cls_label1)

            # train Gaussian2
            feat2 = rep2[:args.labeled_bs]  # fearure
            mask2 = label_batch[:args.labeled_bs] # mask
            cls_label2 = torch.stack([torch.arange(num_classes)] * args.labeled_bs).cuda()  # class label
            cur_cls_label2 = tools.build_cur_cls_label(mask2, num_classes)
            pred_cl2 = tools.clean_mask(outputs2[:args.labeled_bs], cls_label2, True)
            vecs2, proto_loss2, = tools.cal_protypes(feat2, mask2, num_classes)
            res2 = tools.GMM(feat2, vecs2, pred_cl2, mask2, cur_cls_label2)
            gmm_loss2 = tools.cal_gmm_loss(outputs_soft2[:args.labeled_bs], res2, cur_cls_label2, mask2) + proto_loss2

            # Gaussian2 predict
            feat2_u = rep2[args.labeled_bs:] # fearure
            pred_cl2_u = tools.clean_mask_predict(outputs2[args.labeled_bs:], True)
            res2_u = tools.GMM_predict(feat2_u, vecs2, pred_cl2_u)
            gmm_loss2_u = tools.gmm_loss(outputs_soft2[args.labeled_bs:], res2_u, cur_cls_label2)

            ## CGMM consistency_loss
            res1_soft = torch.softmax(res1, dim=1)
            res2_soft = torch.softmax(res2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss = consistency_weight *torch.mean((res1_soft - res2_soft)**2)

            # kt-cps
            weight_A = distdw.get_ema_weights(outputs1[:args.labeled_bs].detach(), label_batch[:args.labeled_bs].detach())
            weight_B = distdw.get_ema_weights(outputs2[:args.labeled_bs].detach(),label_batch[:args.labeled_bs].detach())

            weight_A = weight_A.cpu().numpy()
            weight_B = weight_B.cpu().numpy()

            unsup_loss_func_A = RobustCrossEntropyLoss(weight=weight_A)
            unsup_loss_func_B = RobustCrossEntropyLoss(weight=weight_B)

            max_A = torch.argmax(outputs1.detach(), dim=1, keepdim=True).long()
            max_B = torch.argmax(outputs2.detach(), dim=1, keepdim=True).long()
            loss_cps = unsup_loss_func_B(outputs1, max_B) + unsup_loss_func_A(outputs2, max_A)

            # all loss
            model1_loss = loss1 +  args.lanmuda*(gmm_loss1 + gmm_loss1_u + consistency_loss)
            model2_loss = loss2 +  args.lanmuda*(gmm_loss2 + gmm_loss2_u + consistency_loss)
            loss = model1_loss + model2_loss + consistency_weight*loss_cps

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/gmm_loss1',
                              gmm_loss1, iter_num)
            writer.add_scalar('loss/gmm_loss2',
                              gmm_loss2, iter_num)
            writer.add_scalar('loss/gmm_loss1_u',
                              gmm_loss1_u, iter_num)
            writer.add_scalar('loss/gmm_loss2_u',
                              gmm_loss2_u, iter_num)
            writer.add_scalar('loss/loss1',
                              loss1, iter_num)
            writer.add_scalar('loss/loss2',
                              loss2, iter_num)
            writer.add_scalar('loss/proto_loss1',
                              proto_loss1, iter_num)
            writer.add_scalar('loss/proto_loss2',
                              proto_loss2, iter_num)
            writer.add_scalar('loss/loss_cps',
                              loss_cps, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            

            if iter_num > max_iterations * 0.5 and iter_num % 200 == 0:
                model1.eval()
                ############## model1_test ###############################
                avg_metric_test = test_all_case(
                    model1, args.root_path, test_list="test.txt", num_classes=14,
                    patch_size=(64, 160, 160), stride_xy=80, stride_z=32)
                if avg_metric_test[:, 0].mean() > best_performance1_test:
                    best_performance1_test = avg_metric_test[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_test_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1_test, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_test_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar('test_model_1/model1_dice_score',
                                  avg_metric_test[:, 0].mean(), iter_num)
                writer.add_scalar('test_model_1/model1_dice_score1',
                                  avg_metric_test[0, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score2',
                                  avg_metric_test[1, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score3',
                                  avg_metric_test[2, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score4',
                                  avg_metric_test[3, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score5',
                                  avg_metric_test[4, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score6',
                                  avg_metric_test[5, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score7',
                                  avg_metric_test[6, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score8',
                                  avg_metric_test[7, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score9',
                                  avg_metric_test[8, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score10',
                                  avg_metric_test[9, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score11',
                                  avg_metric_test[10, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score12',
                                  avg_metric_test[11, 0], iter_num)
                writer.add_scalar('test_model_1/model1_dice_score13',
                                  avg_metric_test[12, 0], iter_num)

                writer.add_scalar('test_model_1/model1_hd95',
                                  avg_metric_test[0, 1].mean(), iter_num)
                logging.info(
                    'iteration %d : model1_test_dice_score : %f model1_test_hd95 : %f' % (
                        iter_num, avg_metric_test[0, 0].mean(), avg_metric_test[0, 1].mean()))

                model1.train()

                model2.eval()
                ############## model2_test ###############################
                avg_metric_test2 = test_all_case(
                    model2, args.root_path, test_list="test.txt", num_classes=14,
                    patch_size=(64, 160, 160), stride_xy=80, stride_z=32)
                if avg_metric_test2[:, 0].mean() > best_performance2_test:
                    best_performance2_test = avg_metric_test2[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_test_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2_test, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_test_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                writer.add_scalar('test_model_2/model2_dice_score',
                                  avg_metric_test2[:, 0].mean(), iter_num)
                writer.add_scalar('test_model_2/model2_dice_score1',
                                  avg_metric_test2[0, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score2',
                                  avg_metric_test2[1, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score3',
                                  avg_metric_test2[2, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score4',
                                  avg_metric_test2[3, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score5',
                                  avg_metric_test2[4, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score6',
                                  avg_metric_test2[5, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score7',
                                  avg_metric_test2[6, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score8',
                                  avg_metric_test2[7, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score9',
                                  avg_metric_test2[8, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score10',
                                  avg_metric_test2[9, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score11',
                                  avg_metric_test2[10, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score12',
                                  avg_metric_test2[11, 0], iter_num)
                writer.add_scalar('test_model_2/model2_dice_score13',
                                  avg_metric_test2[12, 0], iter_num)

                writer.add_scalar('test_model_2/model2_hd95',
                                  avg_metric_test2[0, 1].mean(), iter_num)

                logging.info(
                    'iteration %d : model2_test_dice_score : %f model2_test_hd95 : %f' % (
                        iter_num, avg_metric_test2[0, 0].mean(), avg_metric_test2[0, 1].mean()))

                model2.train()


            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    end_time = time.time()
    total_time = end_time - start_time  
    print(f"Model TIMES: {total_time}")

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "guidednet_flare_2024/{}_{}_{}".format(args.exp, args.model, args.data_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
