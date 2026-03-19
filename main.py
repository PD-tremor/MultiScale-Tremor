#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py (PURE ViT, Flow-magnitude) for PD tremor assessment
"""

import os
import time
import json
import shutil
import math
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import torchvision
from torch.utils.data import WeightedRandomSampler

from ops.dataset import TSNDataSet
from ops.models_flow_m import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy

from tensorboardX import SummaryWriter


# ===================== Global config =====================
best_prec1 = 0.0
ACCEPTABLE_MARGIN = 1
# =========================================================


# ------------------------- Utils: class stats from list files -------------------------
def infer_num_classes_from_lists(*list_files) -> Tuple[Optional[int], List[int]]:
    labels = []
    for lf in list_files:
        if lf is None or (not os.path.isfile(lf)):
            continue
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    labels.append(int(parts[2]))
                except Exception:
                    continue
    if not labels:
        return None, []
    uniq = sorted(set(labels))
    num_class = max(uniq) + 1
    return num_class, uniq


def parse_labels_from_list(list_file: str) -> List[int]:
    """Read labels (3rd column) from TSN list file: 'path num_frames label'."""
    labels = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            labels.append(int(parts[2]))
    return labels


def compute_class_counts_and_weights(
    train_list: str,
    num_class: int,
    device: str = "cuda",
    mode: str = "inv_sqrt",  # "inv" or "inv_sqrt"
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:

    labels = parse_labels_from_list(train_list)
    cnt = torch.zeros(num_class, dtype=torch.long)
    for y in labels:
        if 0 <= y < num_class:
            cnt[y] += 1
    cnt = cnt.clamp_min(1)

    if mode == "inv":
        w = 1.0 / cnt.float()
    else:
        w = 1.0 / torch.sqrt(cnt.float())

    w = w / w.mean()  # normalize to mean=1 (keeps LR scale stable)
    return cnt.to(device), w.to(device), labels


# ------------------------- Figures & checkpoint -------------------------
def _ensure_fig_dir():
    fig_dir = os.path.join(args.root_log, args.store_name, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def save_curves(history):
    fig_dir = _ensure_fig_dir()

    plt.figure()
    plt.plot(history["epochs"], history["train_loss"], label="train_loss")
    plt.plot(history["epochs"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["epochs"], history["val_top1"], label="val_top1")
    plt.plot(history["epochs"], history["val_accept"], label=f"val_accept@{ACCEPTABLE_MARGIN}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "val_acc_curves.png"), dpi=200)
    plt.close()

    with open(os.path.join(fig_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def save_confusion_matrix(y_true, y_pred, num_class, tag="best"):
    fig_dir = _ensure_fig_dir()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_class)))
    np.save(os.path.join(fig_dir, f"confusion_matrix_{tag}.npy"), cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar()
    tick_marks = np.arange(num_class)
    plt.xticks(tick_marks, [str(i) for i in range(num_class)], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(num_class)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"confusion_matrix_{tag}.png"), dpi=200)
    plt.close()


def save_multiclass_roc(y_true, y_score, num_class, tag="best"):
    fig_dir = _ensure_fig_dir()
    y_true_bin = label_binarize(y_true, classes=list(range(num_class)))

    plt.figure()
    any_plotted = False

    for c in range(num_class):
        if y_true_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_score[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"class {c} (AUC={roc_auc:.3f})")
        any_plotted = True

    if y_true_bin.sum() > 0:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, linestyle=":", linewidth=2, label=f"micro (AUC={auc_micro:.3f})")
        any_plotted = True

    if not any_plotted:
        return

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({tag})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"roc_{tag}.png"), dpi=200)
    plt.close()

    np.save(os.path.join(fig_dir, f"roc_scores_{tag}.npy"), y_score)


def check_rootfolders():
    folders_util = [
        args.root_log,
        args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name),
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder, exist_ok=True)


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))


# ------------------------- Optimizer & LR schedule (AdamW + warmup cosine) -------------------------
def build_adamw(model, base_lr, weight_decay):
    """
    Create AdamW param groups:
    - decay: Linear/Conv weights
    - no_decay: bias, LayerNorm/BatchNorm/GroupNorm weights
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm = (".norm" in name.lower()) or ("layernorm" in name.lower()) or ("bn" in name.lower())
        if is_bias or is_norm or p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)

    print(f"[OPT] AdamW param groups: 2")
    print(f"[OPT] decay params: {sum(p.numel() for p in decay):,} | no_decay params: {sum(p.numel() for p in no_decay):,}")

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def warmup_cosine_lr(epoch, step_in_epoch, steps_per_epoch, base_lr, min_lr, warmup_epochs, total_epochs):
    global_step = epoch * steps_per_epoch + step_in_epoch
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))

    if global_step < warmup_steps:
        return base_lr * float(global_step + 1) / float(warmup_steps)

    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ------------------------- Logit adjustment (prior bias) -------------------------
def compute_logit_bias_from_counts(class_counts: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    logit_bias = tau * log(p_c)
    Helps avoid class collapse (especially in ordinal labels where middle class is squeezed).
    """
    prior = (class_counts.float() / class_counts.float().sum()).clamp_min(1e-6)
    return (float(tau) * prior.log())


# ------------------------- Train / Val -------------------------
def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, num_class, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        lr = warmup_cosine_lr(
            epoch=epoch,
            step_in_epoch=i,
            steps_per_epoch=len(train_loader),
            base_lr=args.lr,
            min_lr=getattr(args, "min_lr", 0.0),
            warmup_epochs=getattr(args, "warmup_epochs", 0),
            total_epochs=args.epochs,
        )
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(input)
                if args.logit_adjust and hasattr(args, "_logit_bias") and args._logit_bias is not None:
                    output = output + args._logit_bias
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            if args.clip_gradient is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.clip_gradient)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input)
            if args.logit_adjust and hasattr(args, "_logit_bias") and args._logit_bias is not None:
                output = output + args._logit_bias
            loss = criterion(output, target)
            loss.backward()
            if args.clip_gradient is not None:
                clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()

        k2 = min(5, num_class)
        prec1, preck = accuracy(output.data, target, topk=(1, k2))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        topk.update(preck.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            msg = (
                f"Epoch: [{epoch}][{i}/{len(train_loader)}], lr: {lr:.6e}\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Prec@{k2} {topk.val:.3f} ({topk.avg:.3f})"
            )
            print(msg)
            if log is not None:
                log.write(msg + "\n")
                log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar("loss/train", losses.avg, epoch)
        tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)
        tf_writer.add_scalar("acc/train_topk", topk.avg, epoch)
        tf_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    return losses.avg


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None, num_class=5, return_details=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk_meter = AverageMeter()
    acceptable = AverageMeter()

    all_preds = []
    all_labels = []
    all_scores = []

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            output = model(input)
            if args.logit_adjust and hasattr(args, "_logit_bias") and args._logit_bias is not None:
                output = output + args._logit_bias

            loss = criterion(output, target)

            pred = output.argmax(dim=1)
            ok = (pred - target).abs().le(ACCEPTABLE_MARGIN)
            acc_accept = ok.float().mean().item() * 100.0
            acceptable.update(acc_accept, input.size(0))

            k2 = min(5, num_class)
            prec1, preck = accuracy(output.data, target, topk=(1, k2))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            topk_meter.update(preck.item(), input.size(0))

            all_preds.extend(pred.detach().cpu().tolist())
            all_labels.extend(target.detach().cpu().tolist())
            all_scores.append(output.detach().cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                msg = (
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Prec@{k2} {topk_meter.val:.3f} ({topk_meter.avg:.3f})\t"
                    f"Accept@{ACCEPTABLE_MARGIN} {acceptable.val:.3f} ({acceptable.avg:.3f})"
                )
                print(msg)
                if log is not None:
                    log.write(msg + "\n")
                    log.flush()

    # ---- debug histogram: true vs pred ----
    true_hist = np.bincount(np.array(all_labels, dtype=np.int64), minlength=num_class)
    pred_hist = np.bincount(np.array(all_preds, dtype=np.int64), minlength=num_class)
    print("[VAL] True hist :", true_hist.tolist())
    print("[VAL] Pred hist :", pred_hist.tolist())

    msg = (
        f"Testing Results: Prec@1 {top1.avg:.3f} Prec@{min(5, num_class)} {topk_meter.avg:.3f} "
        f"Accept@{ACCEPTABLE_MARGIN} {acceptable.avg:.3f} Loss {losses.avg:.5f}"
    )
    print(msg)

    if log is not None:
        log.write(msg + "\n")
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar("loss/val", losses.avg, epoch)
        tf_writer.add_scalar("acc/val_top1", top1.avg, epoch)
        tf_writer.add_scalar("acc/val_topk", topk_meter.avg, epoch)
        tf_writer.add_scalar(f"acc/val_accept_margin_{ACCEPTABLE_MARGIN}", acceptable.avg, epoch)

    if return_details:
        all_scores = np.concatenate(all_scores, axis=0) if len(all_scores) else np.zeros((0, num_class))
        return top1.avg, losses.avg, acceptable.avg, all_labels, all_preds, all_scores
    return top1.avg


# ------------------------- Main -------------------------
def main():
    global args, best_prec1
    args = parser.parse_args()

    # ---- add robust defaults (won't break if opts.py doesn't define these args) ----
    args.balance_sampler = bool(getattr(args, "balance_sampler", False))  # default OFF
    args.class_weight_mode = str(getattr(args, "class_weight_mode", "inv_sqrt")).lower()  # inv_sqrt recommended
    args.label_smoothing = float(getattr(args, "label_smoothing", 0.0))  # default 0
    args.logit_adjust = bool(getattr(args, "logit_adjust", True))        # default ON
    args.logit_tau = float(getattr(args, "logit_tau", 0.5))              # default 0.5
    args.sampler_power = float(getattr(args, "sampler_power", 0.5))      # 0.5 => sqrt-balanced, 1.0 => inv-balanced

    num_class_cfg, args.train_list, args.val_list, args.root_path, prefix = dataset_config_1fold.return_dataset(
        args.dataset, args.modality
    )

    inferred_num_class, present_labels = infer_num_classes_from_lists(args.train_list, args.val_list)
    if inferred_num_class is not None:
        num_class = inferred_num_class
        print(f"[INFO] Infer num_class from txt: {num_class}, present labels: {present_labels}")
    else:
        num_class = num_class_cfg
        print(f"[WARN] Could not infer from txt, fallback to dataset_config_1fold num_class={num_class}")

    arch_tag = args.arch if hasattr(args, "arch") else "vit"
    args.store_name = "_".join([
        "ViT",
        args.dataset,
        args.modality,
        arch_tag,
        args.consensus_type,
        f"segment{args.num_segments}",
        f"e{args.epochs}",
        args.lr_type if hasattr(args, "lr_type") else "cos",
    ])
    if args.dense_sample:
        args.store_name += "_dense"
    if args.suffix is not None:
        args.store_name += f"_{args.suffix}"

    print("storing name: " + args.store_name)

    check_rootfolders()
    _ensure_fig_dir()

    # ---- build model ----
    model = TSN(
        num_class=num_class,
        num_segments=args.num_segments,
        modality=args.modality,
        base_model=arch_tag,
        consensus_type=args.consensus_type,
        dropout=args.dropout,
        img_feature_dim=args.img_feature_dim,
        partial_bn=False,
        pretrain=args.pretrain,
        is_shift=False,
        temporal_pool=False,
        non_local=False,
        # lock important behavior
        head_norm="gn",
        magnitude_log1p=True,
        magnitude_frame_norm="none",
    )

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = build_adamw(model, base_lr=args.lr, weight_decay=args.weight_decay)

    # ---- AMP ----
    use_amp = bool(getattr(args, "amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume ----
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint.get("epoch", 0)
            best_prec1 = float(checkpoint.get("best_prec1", 0.0))
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', -1)})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # ---- tune_from ----
    if args.tune_from:
        print(f"=> fine-tuning from '{args.tune_from}'")
        sd = torch.load(args.tune_from, map_location="cpu")
        sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd

        sd2 = {}
        for k, v in sd.items():
            kk = k[len("module."):] if k.startswith("module.") else k
            sd2[kk] = v

        if args.dataset not in args.tune_from:
            sd2 = {k: v for k, v in sd2.items() if "classifier" not in k and "new_fc" not in k}

        missing, unexpected = model.load_state_dict(sd2, strict=False)
        print(f"[tune_from] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    cudnn.benchmark = True

    # Data loading
    normalize = GroupNormalize(input_mean, input_std)

    if args.modality == "RGB":
        data_length = 1
    elif args.modality in ["Flow", "RGBDiff"]:
        data_length = 5
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    # mild jitter (safe for hand-in-center videos): small multi-scale crop + optional flip
    do_flip = not ("something" in args.dataset or "jester" in args.dataset)
    train_transform = torchvision.transforms.Compose([
        GroupMultiScaleCrop(crop_size, [1.0, 0.95, 0.9]),
        GroupRandomHorizontalFlip(is_flow=False) if do_flip else IdentityTransform(),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
    ])

    val_transform = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
    ])

    train_dataset = TSNDataSet(
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        transform=train_transform,
        dense_sample=args.dense_sample,
    )

    val_dataset = TSNDataSet(
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=val_transform,
        dense_sample=args.dense_sample,
    )

    # ---- class weights from train_list ----
    class_counts, class_weights, train_labels = compute_class_counts_and_weights(
        args.train_list, num_class=num_class, device="cuda", mode=args.class_weight_mode
    )
    val_labels = parse_labels_from_list(args.val_list)
    val_cnt = np.bincount(val_labels, minlength=num_class)

    print("[DATA] Train class counts:", class_counts.detach().cpu().tolist())
    print("[DATA] Val   class counts:", val_cnt.tolist())
    print(f"[DATA] Class weights(mode={args.class_weight_mode}, mean=1):", [float(x) for x in class_weights.detach().cpu()])

    # ---- logit adjustment bias ----
    args._logit_bias = None
    if args.logit_adjust:
        args._logit_bias = compute_logit_bias_from_counts(class_counts, tau=args.logit_tau).cuda()
        print(f"[LOSS] LogitAdjust ON (tau={args.logit_tau})")
    else:
        print("[LOSS] LogitAdjust OFF")

    # ---- optional sqrt-balanced sampler (gentle) ----
    train_sampler = None
    if args.balance_sampler:
        if len(train_labels) != len(train_dataset):
            print(f"[WARN] train_labels({len(train_labels)}) != len(train_dataset)({len(train_dataset)}). sampler disabled.")
        else:
            cnt_np = class_counts.detach().cpu().numpy().astype(np.float64)
            cnt_np = np.maximum(cnt_np, 1.0)
            # sampler_power=0.5 => 1/sqrt(n); 1.0 => 1/n
            p = float(args.sampler_power)
            sample_w = np.array([1.0 / (cnt_np[y] ** p) for y in train_labels], dtype=np.float64)
            train_sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_w),
                num_samples=len(sample_w),
                replacement=True
            )
            print(f"[DATA] Using WeightedRandomSampler ON (power={p}).")
    else:
        print("[DATA] WeightedRandomSampler OFF (shuffle=True).")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ---- Loss: Weighted CrossEntropy (stable) ----
    if args.loss_type == "nll":
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(args.label_smoothing),
        ).cuda()
        print(f"[LOSS] Weighted CE (label_smoothing={args.label_smoothing})")
    else:
        raise ValueError("Unknown loss type")

    if args.evaluate:
        validate(val_loader, model, criterion, 0, log=None, tf_writer=None, num_class=num_class, return_details=False)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, "log.csv"), "w")
    with open(os.path.join(args.root_log, args.store_name, "args.txt"), "w") as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    history = {"epochs": [], "train_loss": [], "val_loss": [], "val_top1": [], "val_accept": []}
    best_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch, log_training, tf_writer,
            num_class=num_class, scaler=scaler if use_amp else None
        )

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, val_loss, val_accept, y_true, y_pred, y_score = validate(
                val_loader, model, criterion, epoch, log_training, tf_writer,
                num_class=num_class, return_details=True
            )

            history["epochs"].append(int(epoch))
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_top1"].append(float(prec1))
            history["val_accept"].append(float(val_accept))
            save_curves(history)

            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = float(prec1)
                best_epoch = int(epoch)
                save_confusion_matrix(y_true, y_pred, num_class=num_class, tag="best")
                save_multiclass_roc(y_true, y_score, num_class=num_class, tag="best")

            tf_writer.add_scalar("acc/val_top1_best", best_prec1, epoch)

            best_msg = f"Best Prec@1: {best_prec1:.3f} (epoch {best_epoch})"
            print(best_msg)
            log_training.write(best_msg + "\n")
            log_training.flush()

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": arch_tag,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_prec1": best_prec1,
                    "num_class": num_class,
                },
                is_best,
            )


if __name__ == "__main__":
    main()

