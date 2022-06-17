import collections
import json

from PIL import Image
import operator
import argparse
import os
import os.path as osp
import shutil
import time
import numpy as np
import copy
import logging
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
# from torchvision.models import resnet50
from resnet_nohead import resnet50
# used for logging to TensorBoard
from helper import LossTracker, ImageClassifier
from util import TensorboardLogger

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--rounds', default=10, type=int,
                    help='number of total rounds of self training to run')
parser.add_argument('--batch-size', default=64, type=int, dest='batch_size',
                    help='mini-batch size (default: 64) for training')
# parser.add_argument('--test-batch-size', default=256, type=int, dest='test_batch_size',
#                     help='mini-batch size (default: 256) for testing')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=50, type=int,
                    help='total number of layers (default: 101), only 152,101,50,18 are supported')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Office31A2W', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard', default=True, type=bool,
                    help='Log progress to TensorBoard')

# model training
parser.add_argument('--src-root', default='/home/data/office31/amazon/', type=str, dest='src_root',
                    help='address of source data root folder')
parser.add_argument('--src-train-list', default='/home/data/office31/amazon/image.trainlist.txt', type=str, dest='src_train_list',
                    help='the source image_label list for training, which can be changed in terms of the item labels (not the labels)')
parser.add_argument('--src-gt-list', default='/home/data/office31/amazon/image.list.txt', type=str, dest='src_gt_list',
                    help='the source image_label list for evaluation, which are not changed')

parser.add_argument('--tgt-root', default='/home/data/office31/webcam/', type=str, dest='tgt_root',
                    help='address of target data root folder')
parser.add_argument('--tgt-train-list', default='/home/data/office31/webcam/image.trainlist.txt', type=str, dest='tgt_train_list',
                    help='the target image_label list in training/self-training process, which may be updated dynamically')
parser.add_argument('--tgt-gt-list', default='/home/data/office31/webcam/image.list.txt', type=str, dest='tgt_gt_list',
                    help='the target image_label list for evaluation, which are not changed')

# cbst set reg-weights to all 0
# crst reg-weights
parser.add_argument('--mr-weight-l2', default=0., type=float, dest='mr_weight_l2',
                    help='weight of l2 model regularization')
parser.add_argument('--mr-weight-ent', default=0., type=float, dest='mr_weight_ent',
                    help='weight of negative entropy model regularization')
parser.add_argument('--mr-weight-kld', default=0., type=float, dest='mr_weight_kld',
                    help='weight of kld model regularization')
parser.add_argument('--mr-weight-src', default=0., type=float, dest='mr_weight_src',
                    help='weight of source model regularization')

parser.add_argument('--ls-weight-l2', default=0., type=float, dest='ls_weight_l2',
                    help='weight of l2 label smoothing')
parser.add_argument('--ls-weight-negent', default=0., type=float, dest='ls_weight_negent',
                    help='weight of negative entropy label smoothing')
# parser.add_argument('--lblsmo-weight-kld', default=0., type=float, dest='ls_weight_kld',
#                     help='weight of kld label smoothing')

parser.add_argument('--num-classes', default=31, type=int, dest='num_classes',
                    help='the number of classes')
parser.add_argument('--gpus', default='0', type=str,
                    help='the number of classes')
parser.add_argument('--mode', default='cbst', type=str,
                    help='training mode')
# self-trained network
parser.add_argument('--kc-policy', default='global', type=str, dest='kc_policy',
                    help='The policy to determine kc. Valid values: "global" for global threshold,'
                         ' "cb" for class-balanced threshold, "rcb" for reweighted class-balanced threshold')
parser.add_argument('--kc-value', default='conf', type=str,
                    help='The way to determine kc values, either "conf", or "prob".')
parser.add_argument('--init-tgt-port', default=0.3, type=float, dest='init_tgt_port',
                    help='The initial portion of target to determine kc')
parser.add_argument('--max-tgt-port', default=0.6, type=float, dest='max_tgt_port',
                    help='The max portion of target to determine kc')
parser.add_argument('--tgt-port-step', default=0.05, type=float, dest='tgt_port_step',
                    help='The portion step in target domain in every round of self-paced self-trained neural network')
parser.add_argument('--init-src-port', default=0.5, type=float, dest='init_src_port',
                    help='The initial portion of source portion for self-trained neural network')
parser.add_argument('--max-src-port', default=0.8, type=float, dest='max_src_port',
                    help='The max portion of source portion for self-trained neural network')
parser.add_argument('--src-port-step', default=0.05, type=float, dest='src_port_step',
                    help='The portion step in source domain in every round of self-paced self-trai152ned neural network')
parser.add_argument('--init-randseed', default=0, type=int, dest='init_randseed',
                    help='The initial random seed for source selection')
parser.add_argument('--lr-stepsize', default=7, type=int, dest='lr_stepsize',
                    help='The step size of lr_stepScheduler')
parser.add_argument('--lr-stepfactor', default=0.1, type=float, dest='lr_stepfactor',
                    help='The step factor of lr_stepScheduler')
parser.add_argument('--mine-port', default=0.01, type=float, dest='mine_port',
                    help='minimum port to identify rare class')
parser.add_argument('--rare-cls-num', default=10, type=int, dest='rare_cls_num',
                    help='minimum number of samples to identify rare class')
parser.add_argument('--save-ckpt', default=False, type=bool, dest='save_ckpt',
                    help='save checkpoint including optimizer, model and round index')
parser.add_argument('--conf-thresh', default=0.90, type=float, dest='conf_thresh',
                    help='confidence threshold of different classes')


parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
curr_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

args = parser.parse_args()
print(args)
mr_weight_ent = args.mr_weight_ent
mr_weight_kld = args.mr_weight_kld
device = "cuda" if torch.cuda.is_available() else "cpu"
reg_weight_tgt = 0.1
mode = args.mode

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

def flip(img):
    '''flip horizontally'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def main():
    if args.tensorboard:
        tensorboard_logger = TensorboardLogger("/home/crstLog/Office31A2W/%s"%(curr_time))

    seed_torch(args.init_randseed)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    with open(args.src_gt_list, 'r') as f:
        src_gt_list = f.readlines()
    args.num_src = len(src_gt_list)

    # Office31
    root = "/home/data/Office31"
    source = "A"
    target = 'W'
    task_map = {"A": "amazon", "W": "webcam", "D": "dslr"}
    tgt_gt_set = ImageDataset(txt_file=args.tgt_gt_list, root_dir=root+"/"+task_map[target]+"/images", num_classes=31, reg_weight = 0.0, transform=data_transforms['val'])

    # VisDA2017
    # visDA17_valset = ImageDataset(txt_file=args.tgt_gt_list, root_dir=args.tgt_root, reg_weight = 0.0, transform=data_transforms['val'])
    tgt_gt_loader = torch.utils.data.DataLoader(tgt_gt_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # create model
    # pretrained_model = resnet50(False, num_classes=31)
    # state_dict = torch.load("/home/checkpoint/office31/best_resnet_amazon.pth")
    # pretrained_model.load_state_dict(state_dict)

    # DANN checkpoint
    pretrained_model = resnet50(False)
    pretrained_model = ImageClassifier(pretrained_model, 31, bottleneck_dim=256, pool_layer=None)
    state_dict = torch.load('/home/dannLog/Office31_A2W/elite/checkpoints/A_W_0.pth')
    pretrained_model.load_state_dict(state_dict)

    device = torch.device("cuda:"+args.gpus)
    model = pretrained_model.to(device)
    print("model setup successful")

    # all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr,
    #                         momentum=args.momentum, nesterov=True,
    #                         weight_decay=args.weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.lr_stepfactor)
    tgt_portion = args.init_tgt_port
    src_portion = args.init_src_port
    randseed = args.init_randseed

    # oracle tgt distribution
    # tgt_class_size = np.load("/home/data/office31/webcam/class_size.npy")
    src_train_list = '/home/data/office31/amazon/image.trainlist.txt'
    src_gt_list = '/home/data/office31/amazon/image.list.txt'
    tgt_gt_list = '/home/data/office31/webcam/image.list.txt'

    pseudo_label_path = "/home/crstLog/Office31A2W/%s/pseudo_labels/" % (curr_time)
    sel_src_path = "/home/crstLog/Office31A2W/%s/sel_src/" % (curr_time)
    save_stats_path = "/home/crstLog/Office31A2W/%s/stats/" % (curr_time)
    checkpoint_path = "/home/crstLog/Office31A2W/%s/checkpoint/" % (curr_time)
    os.makedirs(pseudo_label_path, exist_ok=True)
    os.makedirs(sel_src_path, exist_ok=True)
    os.makedirs(save_stats_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    best_acc = 0
    history = collections.defaultdict(list)
    for round_idx in range(args.rounds):

        # evaluate on validation set
        # confidence vectors for target
        conf_dict, pred_cls_num, prob_tensor, predictions, imageNameList, loss, val_acc = validate(tgt_gt_loader, model, round_idx, tensorboard_logger, pseudo_label_path)
        print("Validation round: %s \t loss: %.2f\t Acc: %.2f"%(round_idx, loss, val_acc))
        tensorboard_logger.update(step=round_idx, val_loss=loss)
        tensorboard_logger.update(step=round_idx, val_acc=val_acc)

        history['val_acc'].append(float(val_acc))
        np.save('/home/crstLog/Office31A2W/%s/stats/%s_tgt_dist.npy'%(curr_time, round_idx), pred_cls_num)
        # generate kct
        # cls_thresholds = kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path)
        cls_thresholds = np.ones(args.num_classes) * args.conf_thresh
        np.save(osp.join(save_stats_path, 'class_thresholds.%s.npy'%round_idx), cls_thresholds)
        # kct_matrix = kc_parameters(tgt_portion, args.kc_policy, pred_logit_list, num_class, device)
        # angle_matrix = kc_parameters(tgt_portion, args.kc_policy, angle_list, num_class, device)
        # pred_softlabel_matrix = soft_pseudo_label(pred_logit_list, kct_matrix, args)
        # angle_softlabel_matrix = soft_pseudo_label(angle_list, angle_matrix, args)
        tgt_pseudo_label_list = label_selection(prob_tensor, cls_thresholds, pred_cls_num, imageNameList, pseudo_label_path, round_idx, soft_pseudo_lable=False)

        # select part of source data for model retraining
        src_sub_train_list = sel_src_path + '%s_train_list.txt'%round_idx
        num_src_sel, src_sub_train_list = saveSRCtxt(src_gt_list, src_sub_train_list, src_portion, args.num_src, randseed)
        tensorboard_logger.update(step=round_idx, num_sel_source=num_src_sel)
        print('round %s: select %s of samples from src domain' % (round_idx, num_src_sel))

        # update random seed
        randseed = randseed + 1
        # update next round's src portion and tgt portion
        src_portion = min(src_portion + args.src_port_step,args.max_src_port)
        tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)

        # train for one epoch
        # src domain ( add sample difficulty in the future)
        src_train_set = ImageDataset(txt_file=src_sub_train_list, root_dir=args.src_root,reg_weight=0, num_classes=31, transform=data_transforms['train'])
        tgt_train_set = ImageDataset(txt_file=tgt_pseudo_label_list, root_dir=args.tgt_root,reg_weight=reg_weight_tgt, num_classes=31, transform=data_transforms['val4mix'])
        mix_trainset = torch.utils.data.ConcatDataset([src_train_set, tgt_train_set])
        mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        model, train_loss, train_accuracy = train(mix_train_loader, model, optimizer, tensorboard_logger)
        tensorboard_logger.update(step=round_idx, train_loss=train_loss)
        tensorboard_logger.update(step=round_idx, train_acc=train_accuracy)

        # remember best precison1 and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if args.save_ckpt:
            save_checkpoint({
            'epoch': round_idx + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'loss': loss
        }, is_best, checkpoint_path,)
        print('Best accuracy:', best_acc)

    with open(osp.join(save_stats_path, "val_acc.json"), 'w') as f:
        json.dump(history, f)

def train(train_loader, model, optimizer, logger):
    """Train for one epoch on the typetraining set"""
    tracker = LossTracker(len(train_loader), 'train:', 1000)
    # switch to train mode
    model.train()

    for i, (image, label, input_name, reg_weight) in enumerate(train_loader):
        label = label.cuda(non_blocking=True)
        image = image.cuda(non_blocking=True)
        reg_weight = reg_weight.cuda(non_blocking=True)

        # compute output
        output = model(image)
        loss = RegCrossEntropyLoss(i, output, label, reg_weight, mode=mode, logger=logger)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(loss, output, label)
    return model, tracker.losses.avg, tracker.top1.avg


def validate(val_loader, model, round_idx, logger, pseudo_label_path, ):
    """Perform validation on the validation set and save all the confidence vectors"""
    tracker = LossTracker(len(val_loader), "test:", 100)
    # switch to evaluate mode
    model.eval()
    num_classes = val_loader.dataset.num_classes
    conf_dict = {k: [] for k in range(num_classes)}
    prob_tensor = []
    predictions = []
    ImgName_pred_list = []
    pred_class_num = np.zeros(num_classes, dtype=np.int64)  # predicted number of samples for each class
    with torch.no_grad():
        for i, (image, label, input_name, reg_weight) in enumerate(val_loader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            reg_weight = reg_weight.cuda(non_blocking=True)

            logits = model(image)
            # logits_flip = model(flip(image))
            probs = F.softmax(logits, dim=1)
            # probs = F.softmax(logits+logits_flip, dim=1)
            confs, preds = torch.max(probs, dim=1)

            prob_tensor.append(probs)
            predictions.append(preds)

            # labels are just for recording loss and acc
            loss = RegCrossEntropyLoss(i, logits, label, reg_weight, mode="cbst", logger=logger)
            tracker.update(loss, probs, label)

            # different classes have diff number of samples
            for conf, prd, name in zip(confs, preds, input_name):
                prd_cls = int(prd)
                conf_dict[prd_cls].append(float(conf))
                ImgName_pred_list.append('%s\t%s\n'%(name, prd_cls))
                pred_class_num[prd_cls] += 1

    logger.update(step=round_idx, eval_loss=tracker.losses.avg)
    logger.update(step=round_idx, eval_loss=tracker.top1.avg)

    prob_tensor = torch.cat(prob_tensor, dim=0)
    predictions = torch.cat(predictions, dim=0)
    with open(osp.join(pseudo_label_path, "%s.raw.pseudolabels.txt"%round_idx), 'w', encoding='utf-8') as f:
        f.writelines(ImgName_pred_list)

    assert pred_class_num[0] == len(conf_dict[0]), "num of samples in classes should be equal"
    return conf_dict, pred_class_num, prob_tensor, predictions, ImgName_pred_list, tracker.losses.avg, tracker.top1.avg

def kc_parameters(conf_dict, pred_class_num, tgt_portion, round_idx, save_stats_path):
    num_classes = args.num_classes
    kc_thresholds = np.zeros(num_classes) + 1e-8

    for cls_idx in range(num_classes):
        curr_conf = conf_dict[cls_idx]
        num_valid_pred = len(curr_conf)
        if num_valid_pred > 0:
            sorted_curr_conf = sorted(curr_conf, reverse=True)
            curr_tgt_pivot = math.floor(tgt_portion * num_valid_pred) - 1
            kc_thresholds[cls_idx] = sorted_curr_conf[curr_tgt_pivot]
        conf_dict[cls_idx] = None

    # pred class num of tgt domain at this round
    num_mine_id = len(np.nonzero(pred_class_num / np.sum(pred_class_num) < args.mine_port)[0])
    id_all = np.argsort(pred_class_num / np.sum(pred_class_num))
    rare_id = id_all[:args.rare_cls_num]
    mine_id = id_all[:num_mine_id]
    # save mine ids
    np.save(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy', rare_id)
    np.save(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy', mine_id)
    print('Mining ids : {}! {} rarest ids: {}!'.format(mine_id, args.rare_cls_num, rare_id))
    return kc_thresholds

def label_selection(prob_tensor, kc_thresholds, pred_class_num, ImgName_list, pseudo_label_path, round_idx, soft_pseudo_lable):
    batch_size = prob_tensor.shape[0]
    pred_class_num = torch.from_numpy(pred_class_num).to(device)
    pred_class_num = pred_class_num.unsqueeze(0).expand(batch_size, -1)

    prob_tensor *= torch.logical_not(torch.isclose(pred_class_num, torch.zeros(1, dtype=torch.int64, device=device))).float()

    weighted_prob = prob_tensor / torch.from_numpy(kc_thresholds).to(device)
    mask = (weighted_prob > 1).float()
    if not soft_pseudo_lable:
        pseudo_label_prob = prob_tensor * mask
    else:
        soft_pseudo_label = torch.power(weighted_prob, 1.0 / 1.2)  # weighted softmax with temperature
        pseudo_label_prob = soft_pseudo_label * mask
    pseudo_label = torch.argmax(pseudo_label_prob, dim=1).cpu().numpy()

    np.save(osp.join(pseudo_label_path, "%s.confident.pseudolabels.npy"%round_idx), pseudo_label)   # for analysis
    curr_pseudo_label_path = osp.join(pseudo_label_path, "%s.confident.pseudolabels.txt"%round_idx)
    with open(curr_pseudo_label_path, "w") as f:
        for imgLine, pseudo_l in zip(ImgName_list, pseudo_label):
            imgName = imgLine.strip("\n").split("\t")[0]
            f.write("%s\t%s\n"%(imgName, pseudo_l))
    return curr_pseudo_label_path

def RegCrossEntropyLoss(step, outputs, labels, reg_weight, sample_weight=None, mode="cbst", logger=None):
    assert mode in {"cbst", "kld", "lrent"}

    num_classes = args.num_classes
    batch_size = labels.size(0)            # batch_size
    if not sample_weight:
        sample_weight = torch.zeros(batch_size)

    criterion = nn.NLLLoss()
    softmax = F.softmax(outputs, dim=1)   # compute softmax values
    logsoftmax = F.log_softmax(outputs, dim=1)   # compute log softmax values
    labels = labels.squeeze(1)
    loss = criterion(logsoftmax, labels)

    if mode == "cbst":
        return loss
    elif mode == 'kld':
        kld = torch.sum(-logsoftmax / float(num_classes)) * reg_weight / batch_size
        logger.update(step=step, kld=kld*mr_weight_kld)
        logger.update(step=step, ce=loss)
        return loss + kld*mr_weight_kld
    elif mode == "lrent":
        entropy = torch.sum(softmax*logsoftmax)*reg_weight / batch_size
        logger.update(step=step, entropy=entropy*mr_weight_ent)
        logger.update(step=step, ce=loss)
        return loss + entropy * mr_weight_ent


def saveSRCtxt(src_gt_list, src_train_list, src_portion, num_source, randseed):
    with open(src_gt_list, "r") as f:
        item_list = f.readlines()

    num_sel_source = int(np.floor(num_source*src_portion))
    np.random.seed(randseed)

    sel_idx = list( np.random.choice(num_source, num_sel_source, replace=False) )
    item_list = list( operator.itemgetter(*sel_idx)(item_list) )

    with open(src_train_list, 'w') as f:
        f.writelines(item_list)
    return num_sel_source, src_train_list

class ImageDataset(Dataset):

    def __init__(self, txt_file, root_dir, num_classes, reg_weight, transform=transforms.ToTensor()):
        """
        Args:
            txt_fpred_conf_tensorile (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.reg_weight = reg_weight
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        ImgName_Label = str.split(self.images_frame[idx])
        img_name = os.path.join(self.root_dir,ImgName_Label[0])
        img = Image.open(img_name)
        image = img.convert('RGB')
        lbl = np.asarray(ImgName_Label[1:],dtype=np.int64)
        label = torch.from_numpy(lbl)
        reg_weight = torch.from_numpy( np.asarray(self.reg_weight,dtype=np.float32) )

        if self.transform:
            image = self.transform(image)

        return image,label,ImgName_Label[0],reg_weight

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = save_path + 'epoch_'+str(state['epoch']) + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(save_path, 'model_best.pth'))


if __name__ == '__main__':
    main()
