import torch
from knndnn import VGGPD, MLP7, ResNetPD, BasicBlockPD
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from knndnn import knn_predict
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import collections
import numpy as np
import json
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg16
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random
import warnings
import argparse
import os

parser = argparse.ArgumentParser(description='arguments to compute prediction depth for each data sample')
parser.add_argument('--train_ratio', default=0.5, type=float, help='ratio of train split / total data split')
parser.add_argument('--result_dir', default='./cl_results_vgg', type=str, help='directory to save ckpt and results')
parser.add_argument('--data', default='cifar10', type=str, help='dataset')
parser.add_argument('--arch', default='vgg', type=str, help='vgg / mlp / resnet')
parser.add_argument('--get_train_pd', default=True, type=bool, help='get prediction depth for training split')
parser.add_argument('--get_val_pd', default=True, type=bool, help='get prediction depth for validation split')
parser.add_argument('--resume', default=False, type=bool, help='resume from the ckpt')
parser.add_argument('--fraction', default=0.4, type=float, help='ratio of noise')
parser.add_argument('--half', default=False, type=str, help='use amp if GPU memory is 15 GB; set to False if GPU memory is 32 GB ')
parser.add_argument('--num_epochs', default=80, type=int, help='number of epochs for training')
parser.add_argument('--total_iteration', default=15000, type=str, help='if training process is more than total iteration then stop')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--num_samples', default=10000, type=int, help='number of samples')
parser.add_argument('--knn_k', default=30, type=int, help='k nearest neighbors of knn classifier')

args = parser.parse_args()

# hyper parameters
# change cifar10 as (img, label), index
if args.arch == 'mlp':
    'depth index starts from 0 and end with max_prediction_depth - 1'
    max_prediction_depth = 7
elif args.arch == 'vgg':
    max_prediction_depth = 14
elif args.arch == 'resnet':
    max_prediction_depth = 10

lr_init = 0.04
momentum = 0.9
lr_decay = 0.2
if args.arch == 'mlp':
    mile_stones = [1250, 4000, 12000]
elif args.arch == 'vgg':
    mile_stones = [1000, 5000]
elif args.arch == 'resnet':
    mile_stones = [7000]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mile_stone_step(optimizer, curr_iter):
    if curr_iter in mile_stones:
        for param_gp in optimizer.param_groups:
            param_gp['lr'] *= lr_decay


def trainer(trainloader, testloader, model, optimizer, num_epochs, criterion, random_sd, flip):
    curr_iteration = 0
    cos_scheduler = CosineAnnealingLR(optimizer, num_epochs)
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    for epo in range(num_epochs):
        train_acc = 0
        train_num_total = 0
        for (imgs, labels), idx in trainloader:
            curr_iteration += 1
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            logits = model(imgs, train=True)
            loss = criterion(logits, labels)
            prds = logits.argmax(1)
            train_acc += sum(prds == labels)
            train_num_total += imgs.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # mile_stone_step(optimizer, curr_iteration)
        cos_scheduler.step()
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc.item() / train_num_total)
        print('epoch:', epo, 'lr', optimizer.param_groups[0]['lr'], 'loss', loss.item(), 'train_acc',
              train_acc.item() / train_num_total)
        torch.save(model.state_dict(), os.path.join(args.result_dir, 'ms{}_{}sgd{}_{}.pt'.format(args.arch, args.data, random_sd, flip)))
        with torch.no_grad():
            test_acc = 0
            test_num_total = 0
            for (imgs, labels), idx in testloader:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                logits = model(imgs, train=True)
                loss = criterion(logits, labels)
                prds = logits.argmax(1)
                test_acc += sum(prds == labels)
                test_num_total += imgs.shape[0]
        print('epoch:', epo, 'lr', optimizer.param_groups[0]['lr'], 'loss', loss.item(), 'test_acc',
              test_acc.item() / test_num_total)
        history['test_loss'].append(loss.item())
        history['test_acc'].append(test_acc.item() / test_num_total)
        with open(os.path.join(args.result_dir, 'train_test_history_{}_sd{}_{}.pt'.format(args.arch, seed, flip)), 'w') as f:
            json.dump(history, f)

        if curr_iteration >= args.total_iteration:
            break
    return model


def _get_feature_bank_from_kth_layer(model, dataloader, k):
    print(k, 'layer feature bank gotten')
    with torch.no_grad():
        for (img, all_label), idx in dataloader:
            img = img.cuda(non_blocking=True)
            all_label = all_label.cuda(non_blocking=True)
            if args.half:
                with autocast():
                    _, fms = model(img, k, train=False)
            else:
                _, fms = model(img, k, train=False)
    return fms, all_label


def get_knn_prds_k_layer(model, evaloader, floader, k, train_split=True):
    knn_labels_all = []
    knn_conf_gt_all = []  # This statistics can be noisy
    indices_all = []
    f_bank, all_labels = _get_feature_bank_from_kth_layer(model, floader, k)
    f_bank = f_bank.t().contiguous()
    with torch.no_grad():
        for j, ((imgs, labels), idx) in enumerate(evaloader):
            imgs = imgs.cuda(non_blocking=True)
            labels_b = labels.cuda(non_blocking=True)
            nm_cls = 10
            if args.half:
                with autocast():
                    _, inp_f_curr = model(imgs, k, train=False)
            else:
                _, inp_f_curr = model(imgs, k, train=False)
            knn_scores = knn_predict(inp_f_curr, f_bank, all_labels, classes=nm_cls, knn_k=args.knn_k, knn_t=1, rm_top1=train_split)  # B x C
            knn_probs = F.normalize(knn_scores, p=1, dim=1)
            knn_labels_prd = knn_probs.argmax(1)
            knn_conf_gt = knn_probs.gather(dim=1, index=labels_b[:, None])  # B x 1
            knn_labels_all.append(knn_labels_prd)
            knn_conf_gt_all.append(knn_conf_gt)
            indices_all.append(idx)
        knn_labels_all = torch.cat(knn_labels_all, dim=0)  # N x 1
        knn_conf_gt_all = torch.cat(knn_conf_gt_all, dim=0).squeeze()
        indices_all = np.concatenate(indices_all, 0)
    return knn_labels_all, knn_conf_gt_all, indices_all


def _get_prediction_depth(knn_labels_all):
    """
    get prediction depth for a sample. reverse knn labels list and increase the counter until the label is different
    :param knn_labels_all:
    :return:
    """
    pd = 0
    knn_labels_all = list(reversed(knn_labels_all))
    while knn_labels_all[pd] == knn_labels_all[0] and pd <= max_prediction_depth - 2:
        pd += 1
    return max_prediction_depth - pd

def set_seed(seed=1234):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def main(train_idx, val_idx, random_seed=1234, flip=''):
    # for simplicity, we do not use data augmentation when measuring difficulty
    # CIFAR10 w / 40% (Fixed) Randomized Labels
    # only the training dataset is shuffle. Datasets for prediction depth and testing remains the same as cifar10 original
    train_transform = T.Compose([
                                T.RandomCrop(32, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                ])
    test_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                ])
    if args.data == 'cifar10':
        trainset = CIFAR10('./', transform=train_transform, train=False, download=True)
        testset = CIFAR10('./', transform=test_transform, train=True, download=True)
    else:
        raise NotImplementedError

    train_split = Subset(trainset, train_idx)
    supportset = train_split
    val_split = Subset(trainset, val_idx)
    trainloader = DataLoader(train_split, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

    supportloader = DataLoader(supportset, batch_size=len(supportset), shuffle=False, num_workers=1, pin_memory=True)
    if args.get_train_pd:
        # pd (train) data order follows train_indices
        evaluate_loader_train = DataLoader(train_split, batch_size=200, shuffle=False, num_workers=1, pin_memory=True)
    if args.get_val_pd:
        # pd (val) data order follows val_indices
        evaluate_loader_test = DataLoader(val_split, batch_size=200, shuffle=False, num_workers=1, pin_memory=True)

    if args.arch == 'mlp':
        model = MLP7(args.num_classes)
    elif args.arch == 'vgg':
        ecd = vgg16().features
        model = VGGPD(ecd, args.num_classes)
    elif args.arch == 'resnet':
        model = ResNetPD(BasicBlockPD, [2, 2, 2, 2], temp=1.0)
    else:
        raise NotImplementedError

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
    if not args.resume:
        model = trainer(trainloader, testloader, model, optimizer, args.num_epochs, criterion, random_seed, flip)
    else:
        print('loading model from ckpt')
        model.load_state_dict(torch.load(os.path.join(args.result_dir, 'ms{}_{}sgd{}_{}.pt'.format(args.arch, args.data, random_seed, flip))))

    if args.get_train_pd:
        # TODO exclude current batch from support set
        index_knn_y = collections.defaultdict(list)
        index_pd = collections.defaultdict(list)
        knn_gt_conf_all = collections.defaultdict(list)
        for k in range(max_prediction_depth):
            knn_labels, knn_conf_gt_all, indices_all = get_knn_prds_k_layer(model, evaluate_loader_train, supportloader,
                                                                            k, train_split=args.get_train_pd)
            for idx, knn_l, knn_conf_gt in zip(indices_all, knn_labels, knn_conf_gt_all):
                index_knn_y[int(idx)].append(knn_l.item())
                knn_gt_conf_all[int(idx)].append(knn_conf_gt.item())
        for idx, knn_ls in index_knn_y.items():
            index_pd[idx].append(_get_prediction_depth(knn_ls))

        print(len(index_pd), len(index_knn_y), len(knn_gt_conf_all))
        with open(os.path.join(args.result_dir, 'ms{}train_seed{}_f{}_trainpd.pkl'.format(args.arch, random_seed, flip)), 'w') as f:
            json.dump(index_pd, f)

    if args.get_val_pd:
        index_knn_y = collections.defaultdict(list)
        index_pd = collections.defaultdict(list)
        knn_gt_conf_all = collections.defaultdict(list)
        for k in range(max_prediction_depth):
            knn_labels, knn_conf_gt_all, indices_all = get_knn_prds_k_layer(model, evaluate_loader_test, supportloader,
                                                                            k, train_split=not(args.get_val_pd))
            for idx, knn_l, knn_conf_gt in zip(indices_all, knn_labels, knn_conf_gt_all):
                index_knn_y[int(idx)].append(knn_l.item())
                knn_gt_conf_all[int(idx)].append(knn_conf_gt.item())
        for idx, knn_ls in index_knn_y.items():
            index_pd[idx].append(_get_prediction_depth(knn_ls))

        print(len(index_pd), len(index_knn_y), len(knn_gt_conf_all))
        with open(os.path.join(args.result_dir, 'ms{}_seed{}_f{}_test_pd.pkl'.format(args.arch, random_seed, flip)), 'w') as f:
            json.dump(index_pd, f)


if __name__ == '__main__':
    seeds = [9203, 9304, 9837, 9612, 3456, 5210]
    for seed in seeds:
        set_seed(seed)
        train_indices, val_indices = train_test_split(np.arange(args.num_samples), train_size=args.train_ratio,
                                                   test_size=(1 - args.train_ratio))     # split the data
        main(train_indices, val_indices, random_seed=seed, flip='')
        main(val_indices, train_indices, random_seed=seed, flip='flip')
