import argparse
import os
import random
import numpy as np
import torch
import json
import torch.nn as nn
from torch.nn import DataParallel
import collections
import time
from torchvision import datasets, transforms
from tqdm import tqdm
from difficulty import *
from models import AngularNet, Baseline
from utils import LossTracker, AverageMeter
from calibration import calibrationMapping, ece_eval, tace_eval
from timm.data import Mixup
from torch.utils.data import Dataset, DataLoader
import visualize


def main(arg):
    set_seed(arg.seed)
    num_epochs = arg.epochs

    if arg.dst == "cifar10":
        train_ds = datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261)
                    ),
                ]
            ),
            download=True,
        )
        test_ds = datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(36),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261)
                    ),
                ]
            ),
            download=True,
        )

        validation_size = 5000
        train_indices = range(50000)[:-validation_size]
        val_indices = range(50000)[-validation_size:]
        vis_indices = range(1000)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        vis_sampler = torch.utils.data.SubsetRandomSampler(vis_indices)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=args.batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=args.batch_size, sampler=valid_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=100,
        shuffle=False,
    )
    os.makedirs("./figs", exist_ok=True)
    loss_type = arg.loss_type
    history = collections.defaultdict(list)
    print("Training {} model....".format(loss_type))
    net = DataParallel(
        AngularNet(
            num_classes=len(train_loader.dataset.classes),
            loss_type=loss_type,
            arch=arg.arch,
            s=arg.s,
            m=arg.m,
        )
    ).to(device)
    net, calibration_map = train_hyper(
        net,
        num_epochs,
        train_loader,
        valid_loader,
        test_loader,
        loss_type,
        history,
        arg,
    )
    if args.arch == "visualization":
        angular_embeds, angular_labels = get_embeds(net, test_loader)
        visualize.plot3d(
            angular_embeds,
            angular_labels,
            num_classes=10,
            fig_path="./figs/{}.png".format(loss_type),
        )
        print("Saved {} figure".format(loss_type))
        del angular_embeds, angular_labels


def train_hyper(
    model,
    total_epochs,
    train_loader,
    valid_loader,
    test_loader,
    loss_type,
    history,
    arg,
):
    # optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    if arg.aug == "mixup":
        mixup_args = {
            "mixup_alpha": 1.0,
            "cutmix_alpha": 0.0,
            "cutmix_minmax": None,
            "prob": 1.0,
            "switch_prob": 0.0,
            "mode": "batch",
            "label_smoothing": 0,
            "num_classes": 10,
        }
        aug_func = Mixup(**mixup_args)
    else:
        aug_func = nn.Identity()

    calibration_map = None

    step = 0
    for epoch in range(total_epochs):
        tracker = LossTracker(len(train_loader), "step : [{}]".format(step), 1000)
        for i, (b_data, b_labels) in enumerate(train_loader):
            b_data = b_data.cuda(non_blocking=True)
            b_labels = b_labels.cuda(non_blocking=True)
            optimizer.zero_grad()

            if arg.aug == "mixup":
                b_data, b_labels_soft = aug_func(b_data, b_labels)
                logits, _ = model(b_data, b_labels)
                loss = F.cross_entropy(logits, b_labels_soft)
            else:
                b_data = aug_func(b_data)
                logits, _ = model(b_data, b_labels)
                loss = F.cross_entropy(logits, b_labels)

            loss = loss.mean()  # for DataParallel

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            tracker.update(loss, logits, b_labels)
            step += 1
        print(
            "{}:  Epoch [{}/{}], Loss: {:.4f}".format(
                loss_type, epoch + 1, total_epochs, loss.item()
            )
        )
        # scheduler.step()
        history["train_loss"].append(tracker.losses.avg)
        history["train_acc"].append(tracker.top1.avg)
        history["train_top5"].append(tracker.top5.avg)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # calibrate and test
        calibration_map = calibrationMapping(
            10,
            model,
            valid_loader,
            calibration_type=arg.calibration,
            calibration_lr=arg.calib_lr,
        )
        (
            test_acc,
            target_confs,
            output_margins,
            anggaps,
            cosgaps,
            norm_angles,
        ) = test_calibrated_statistics(
            test_loader,
            model,
            calibration_map,
            calibration_type=arg.calibration,
            history=history,
        )

        print("Test accuracy:%s" % test_acc)
        history["acc"].append(test_acc)

        diff_dict = {
            "target_confs": target_confs.tolist(),
            "output_margins": output_margins.tolist(),
            "anggaps": anggaps.tolist(),
            "cosgaps": cosgaps.tolist(),
            "avh": norm_angles.tolist(),
        }
        with open(
            os.path.join(
                arg.result_dir,
                "diff_score{}_{}scale{}sd{}_{}_{}epoch{}.json".format(
                    arg.dst,
                    arg.calibration,
                    arg.arch,
                    arg.s,
                    arg.seed,
                    arg.loss_type,
                    epoch,
                ),
            ),
            "w",
        ) as f:
            json.dump(diff_dict, f)

        torch.save(
            history,
            os.path.join(
                arg.result_dir,
                "{}{}history_loss{}scale{}sd{}{}.pt".format(
                    arg.dst, arg.calibration, arg.arch, arg.s, arg.seed, arg.loss_type
                ),
            ),
        )
        torch.save(
            model.state_dict(),
            os.path.join(
                arg.result_dir,
                "{}{}model{}_scale{}seed{}{}.pt".format(
                    arg.dst, arg.calibration, arg.arch, arg.s, arg.seed, arg.loss_type
                ),
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                arg.result_dir,
                "{}{}optimizer{}_scale{}seed{}{}.pt".format(
                    arg.dst, arg.calibration, arg.arch, arg.s, arg.seed, arg.loss_type
                ),
            ),
        )
        torch.save(
            calibration_map.state_dict(),
            os.path.join(
                arg.result_dir,
                "{}{}calibMap{}_scale{}seed{}{}.pt".format(
                    arg.dst, arg.calibration, arg.arch, arg.s, arg.seed, arg.loss_type
                ),
            ),
        )

    return model, calibration_map


def test_calibrated_statistics(
    testloader,
    model,
    calibration_map=None,
    calibration_type="diagonal_scaling",
    history=None,
):
    num_classes = len(testloader.dataset.classes)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if calibration_type == "diagonal_scaling":
        post = calibration_map.diag.diag()
        post = post.unsqueeze(0)
    elif calibration_type == "temperature_scaling":
        post = calibration_map.temp
        post = post.unsqueeze(0).expand(-1, num_classes)
    elif calibration_type == "matrix_scaling":
        post = calibration_map.mat.weight.diag() - calibration_map.mat.bias
        post = post.unsqueeze(0)
    else:
        raise NotImplementedError

    print("start testing")
    with torch.no_grad():
        # uncalibrated stats
        uncalibrate_loss_tracker = LossTracker(len(testloader), "val", 1000)
        total_uncalibrate_cos_gap = AverageMeter("uncalib_cos_gap", ":.4e")
        total_uncalibrate_ang_gap = AverageMeter("uncalib_ang_gap", ":.4e")
        total_uncalibrate_norm_angle = AverageMeter("uncalib_norm_angle", ":.4e")
        total_uncalibrate_target_conf = AverageMeter("uncalib_target_conf", ":.4e")
        total_uncalibrate_output_margin = AverageMeter("uncalib_output_margin", ":.4e")
        uncalibrate_ece_avg = AverageMeter("uncalibrate ece", ":6.2f")
        uncalibrate_tace_avg = AverageMeter("uncalibrate tace", ":6.2f")

        # calibrated stats
        calibrate_loss_tracker = LossTracker(len(testloader), "val", 1000)
        total_calibrate_cos_gap = AverageMeter("cos_gap", ":.4e")
        total_calibrate_ang_gap = AverageMeter("ang_gap", ":.4e")
        total_calibrate_norm_angle = AverageMeter("norm_angle", ":.4e")
        total_calibrate_target_conf = AverageMeter("target_conf", ":.4e")
        total_calibrate_output_margin = AverageMeter("output_margin", ":.4e")
        calibrate_ece_avg = AverageMeter("calibrate ece", ":6.2f")
        calibrate_tace_avg = AverageMeter("calibrate tace", ":6.2f")

        target_confs = []
        output_margins = []
        anggaps = []
        cosgaps = []
        norm_angles = []
        test_start = time.time()
        for (images, targets) in testloader:
            batch_size = len(images)
            target_onehot = torch.zeros(len(targets), num_classes)
            target_onehot = target_onehot.scatter(
                dim=1, index=targets[:, None], src=torch.ones(len(targets), 1)
            ).to(
                device
            )  # B, C
            images, targets = images.cuda(non_blocking=True), targets.cuda(
                non_blocking=True
            )

            # uncalibrate_statics
            uncalibrated_logits, cos_dists = model(images, targets)
            uncalibrate_loss = criterion(uncalibrated_logits, targets)

            (
                uncalib_target_confidence,
                uncalib_output_margin,
            ) = get_confidence_output_margin(uncalibrated_logits, target_onehot)
            _, uncalib_cos_gap, _, uncalib_ang_gap = angular_gap(
                cos_dists, target_onehot
            )
            uncalib_norm_angle = avh(cos_dists, targets=targets)

            # recode uncalibrated stats
            uncalibrate_loss_tracker.update(
                uncalibrate_loss, uncalibrated_logits, targets
            )
            total_uncalibrate_target_conf.update(
                uncalib_target_confidence.mean(), batch_size
            )
            total_uncalibrate_output_margin.update(
                uncalib_output_margin.mean(), batch_size
            )
            total_uncalibrate_cos_gap.update(uncalib_cos_gap.mean(), batch_size)
            total_uncalibrate_ang_gap.update(uncalib_ang_gap.mean(), batch_size)
            total_uncalibrate_norm_angle.update(uncalib_norm_angle.mean(), batch_size)
            # uncalib_prds_n = torch.softmax(uncalibrated_logits, dim=1).cpu().detach().numpy()
            # np_targets = targets.cpu().numpy()
            # un_ece, _, _, _ = ece_eval(uncalib_prds_n, np_targets)
            # un_tace, _, _, _ = tace_eval(uncalib_prds_n, np_targets)
            # uncalibrate_ece_avg.update(un_ece, np_targets.shape[0])
            # uncalibrate_tace_avg.update(un_tace, np_targets.shape[0])

            # record stats after calibration
            cos_dists_calibrated = post * cos_dists  # B, C
            calibrated_logits = calibration_map(uncalibrated_logits)
            calibrate_loss = criterion(calibrated_logits, targets)

            calib_target_confidence, calib_output_margin = get_confidence_output_margin(
                calibrated_logits, target_onehot
            )
            _, cos_gap, _, ang_gap = angular_gap(
                cos_dists_calibrated, target_onehot, calibration_map.diag.data.diag()
            )
            norm_angle = avh(cos_dists_calibrated, targets=targets)
            # recode calibrated stats
            # calib_prds_n = torch.softmax(calibrated_logits, dim=1).cpu().detach().numpy()
            # ece, _, _, _ = ece_eval(calib_prds_n, np_targets)
            # tace, _, _, _ = tace_eval(calib_prds_n, np_targets)
            # calibrate_ece_avg.update(ece, np_targets.shape[0])
            # calibrate_tace_avg.update(tace, np_targets.shape[0])

            calibrate_loss_tracker.update(calibrate_loss, calibrated_logits, targets)
            total_calibrate_target_conf.update(
                calib_target_confidence.mean(), batch_size
            )
            total_calibrate_output_margin.update(calib_output_margin.mean(), batch_size)
            total_calibrate_cos_gap.update(cos_gap.mean(), batch_size)
            total_calibrate_ang_gap.update(ang_gap.mean(), batch_size)
            total_calibrate_norm_angle.update(norm_angle.mean(), batch_size)

            target_confs.append(calib_target_confidence)
            output_margins.append(calib_output_margin)
            anggaps.append(ang_gap)
            cosgaps.append(cos_gap)
            norm_angles.append(norm_angle)

        test_end = time.time()
        print("testing, elapse time: {}".format(test_end - test_start))
        # test convergence
        # class feature norm
        history["cls_emb_norm"].append(embed_norm(model.module.angular_loss.fc.weight))
        # data feature norm
        emb = model(images, embed=True)
        history["data_emb_norm"].append(embed_norm(emb))
        del emb

        history["uncalibrate_test_loss"].append(uncalibrate_loss_tracker.losses.avg)
        history["uncalibrate_test_acc1"].append(uncalibrate_loss_tracker.top1.avg)
        history["uncalibrate_test_acc5"].append(uncalibrate_loss_tracker.top5.avg)
        history["uncalibrate_cos_gap"].append(total_uncalibrate_cos_gap.avg)
        history["uncalibrate_ang_gap"].append(total_uncalibrate_ang_gap.avg)
        history["uncalibrate_norm_angle"].append(total_uncalibrate_norm_angle.avg)
        history["uncalibrate_target_conf"].append(total_uncalibrate_target_conf.avg)
        history["uncalibrate_output_gap_margin"].append(
            total_uncalibrate_output_margin.avg
        )
        # calibrate
        history["calibrate_test_loss"].append(calibrate_loss_tracker.losses.avg)
        history["calibrate_test_acc1"].append(calibrate_loss_tracker.top1.avg)
        history["calibrate_test_acc5"].append(calibrate_loss_tracker.top5.avg)
        history["calibrate_cos_gap"].append(total_calibrate_cos_gap.avg)
        history["calibrate_ang_gap"].append(total_calibrate_ang_gap.avg)
        history["calibrate_norm_angle"].append(total_calibrate_norm_angle.avg)
        history["calibrate_target_conf"].append(total_calibrate_target_conf.avg)
        history["calibrate_output_gap_margin"].append(total_calibrate_output_margin.avg)

        target_confs = torch.cat(target_confs, 0).cpu().numpy()
        output_margins = torch.cat(output_margins, 0).cpu().numpy()
        anggaps = torch.cat(anggaps, 0).cpu().numpy()
        cosgaps = torch.cat(cosgaps, 0).cpu().numpy()
        norm_angles = torch.cat(norm_angles, 0).cpu().numpy()
        return (
            uncalibrate_loss_tracker.top1.avg,
            target_confs,
            output_margins,
            anggaps,
            cosgaps,
            norm_angles,
        )


def get_calibrated_difficulty_correlation(
    model,
    testloader,
    calibration_map=None,
    calibration_type="diagonal_scaling",
    arg=None,
):
    human_score = np.load(os.path.join("./orders", arg.human_score))

    correlations = {}
    # switch to evaluate mode
    model = model.cuda()
    model.eval()
    num_classes = 10
    criterion = nn.CrossEntropyLoss()
    if calibration_type == "diagonal_scaling":
        post = calibration_map.diag.diag()
    elif calibration_type == "temperature_scaling":
        post = calibration_map.temp
    elif calibration_type == "matrix_scaling":
        post = calibration_map.mat.weight.diag() - calibration_map.mat.bias
    else:
        raise NotImplementedError
    with torch.no_grad():
        target_confs = []
        output_margins = []
        anggaps = []
        cosgaps = []
        norm_angles = []

        for i, (images, target) in enumerate(tqdm(testloader)):
            batch_size = len(images)
            target_onehot = torch.zeros(len(targets), num_classes)
            target_onehot = target_onehot.scatter(
                dim=1, index=targets[:, None], src=torch.ones(len(targets), 1)
            ).to(
                device
            )  # B, C
            images, targets = images.cuda(non_blocking=True), targets.cuda(
                non_blocking=True
            )

            # uncalibrate_statics
            uncalibrated_logits, cos_dists = model(images, targets)
            uncalibrate_loss = criterion(uncalibrated_logits, targets)

            (
                uncalib_target_confidence,
                uncalib_output_margin,
            ) = get_confidence_output_margin(uncalibrated_logits, target_onehot)
            _, uncalib_cos_gap, _, uncalib_ang_gap = angular_gap(
                cos_dists, target_onehot
            )
            uncalib_norm_angle = avh(cos_dists, targets=targets)

            cos_dists_calibrated = post * cos_dists  # B, C
            calibrated_logits = calibration_map(uncalibrated_logits)
            calibrate_loss = criterion(calibrated_logits, targets)

            calib_target_confidence, calib_output_margin = get_confidence_output_margin(
                calibrated_logits, target_onehot
            )
            _, cos_gap, _, ang_gap = angular_gap(
                cos_dists_calibrated, target_onehot, calibration_map.diag.data.diag()
            )
            norm_angle = avh(cos_dists_calibrated, targets=targets)

            target_confs = target_confs.append(calib_target_confidence)
            output_margins = output_margins.append(calib_output_margin)
            anggaps = anggaps.append(ang_gap)
            cosgaps = cosgaps.append(cos_gap)
            norm_angles = norm_angles.append(norm_angle)

        total_target_conf = torch.cat(target_confs, 0).cpu().numpy()
        total_output_margin = torch.cat(output_margins, 0).cpu().numpy()
        total_ang_gap = torch.cat(anggaps, 0).cpu().numpy()
        total_cos_gap = torch.cat(cosgaps, 0).cpu().numpy()
        total_norm_angle = torch.cat(norm_angles, 0).cpu().numpy()

    correlations["target_confidence_spearman"] = get_spearman(
        total_target_conf, human_score
    )
    correlations["target_confidence_tau"] = get_kendalltau(
        total_target_conf, human_score
    )

    correlations["output_margin_spearman"] = get_spearman(
        total_output_margin, human_score
    )
    correlations["output_margin_tau"] = get_kendalltau(total_output_margin, human_score)

    correlations["cos_gap_spearman"] = get_spearman(total_cos_gap, human_score)
    correlations["cos_gap_tau"] = get_kendalltau(total_cos_gap, human_score)

    correlations["ang_gap_spearman"] = get_spearman(total_ang_gap, human_score)
    correlations["ang_gap_tau"] = get_kendalltau(total_ang_gap, human_score)

    correlations["norm_angle_spearman"] = get_spearman(total_norm_angle, human_score)
    correlations["norm_angle_tau"] = get_kendalltau(total_norm_angle, human_score)
    return correlations


def get_embeds(model, loader):
    model = model.to(device).eval()
    full_embeds = []
    full_labels = []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats[:200].to(device)
            full_labels.append(labels[:200])
            embeds = model(feats, embed=True)
            full_embeds.append(F.normalize(embeds, dim=1))
    return torch.cat(full_embeds, 0), torch.cat(full_labels, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AngularGap and Baseline experiments on CIFAR10"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="visualization/resnet18/alexnet/vgg16",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train each model for (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=9999, help="random seed")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--use-cuda", default=True, type=bool, help="enables CUDA training"
    )
    parser.add_argument("--resume", default=False, type=bool, help="resume")
    parser.add_argument("--aug", default="none", type=str, help="none/mixup")
    parser.add_argument(
        "--resume_path",
        default=0,
        type=str,
        help="resume checkpoint from this directory",
    )
    parser.add_argument("--s", default=30.0, type=float, help="scale")
    parser.add_argument("--dst", default="cifar10", type=str, help="train dataset")
    parser.add_argument(
        "--human_score",
        default="cifar10_human_probs.npy",
        type=str,
        help="path to CIFAR10-H human score",
    )
    parser.add_argument(
        "--calibration", default="diagonal_scaling", type=str, help="calibration method"
    )
    parser.add_argument("--result_dir", default="", type=str)
    parser.add_argument("--m", default=0.35, type=float)
    parser.add_argument("--cuda_no", default="0, 1", type=str)
    parser.add_argument("--clip", default=5.0, type=float)
    parser.add_argument(
        "--calib_lr", default=0.01, type=float, help="path to human score"
    )
    parser.add_argument(
        "--loss_type", default="nsl", type=str, help="path to human score"
    )
    parser.add_argument(
        "--num_gpus", default=1, type=int, help="num of gpus for training"
    )
    parser.add_argument("--amp", default=False, type=bool, help="use cuda amp")
    args = parser.parse_args()
    return args


def set_seed(seed=None):
    if seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_no
    set_seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    main(args)
