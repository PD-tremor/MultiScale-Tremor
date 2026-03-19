#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Scale Motion Representation Learning for Video-based Parkinson's Disease Tremor Assessment
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.parallel
import torchvision
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from ops.dataset import TSNDataSet
from ops.models_flow_m import TSN
from ops.transforms import *
from ops import dataset_config


def build_parser():
    p = argparse.ArgumentParser(description="Testing for PURE-ViT Flow-Magnitude TSN")

    p.add_argument('dataset', type=str)

    p.add_argument('--weights', type=str, required=True,
                   help='checkpoint path (comma-separated for ensemble)')
    p.add_argument('--test_segments', type=str, default="8",
                   help='num_segments used for testing (comma-separated if ensemble)')
    p.add_argument('--coeff', type=str, default=None,
                   help='ensemble coeff list, comma-separated')
    p.add_argument('--test_list', type=str, default=None,
                   help='explicit test list file (comma-separated if ensemble)')
    p.add_argument('--csv_file', type=str, default=None)

    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('-j', '--workers', default=8, type=int)

    p.add_argument('--test_crops', type=int, default=1,
                   help='1, 3, 5, 10 (same meaning as original)')
    p.add_argument('--full_res', default=False, action="store_true")
    p.add_argument('--softmax', default=False, action="store_true")

    p.add_argument('--dense_sample', default=False, action="store_true")
    p.add_argument('--twice_sample', default=False, action="store_true")

    p.add_argument('--max_num', type=int, default=-1)
    p.add_argument('--input_size', type=int, default=224)
    p.add_argument('--crop_fusion_type', type=str, default='avg')

    p.add_argument('--gpus', nargs='+', type=int, default=None)

    # keep for compatibility with TSN signature
    p.add_argument('--img_feature_dim', type=int, default=256)
    p.add_argument('--pretrain', type=str, default='none',
                   help="recommend 'none' for testing")

    # explicitly set modality if you ever need (default Flow for your tremor pipeline)
    p.add_argument('--modality', type=str, default='Flow', choices=['Flow', 'RGB'])

    return p


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _normalize_state_dict_keys(sd_raw, model_keys):
    """
    Normalize ckpt keys to match current model naming.
    Keep minimal & safe remaps.
    """
    sd = {}
    for k, v in sd_raw.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]

        # historical rename
        kk = kk.replace(".net.", ".")

        # temporal_module naming alignment (old vs new)
        if kk.startswith("temporal_module."):
            cand = kk.replace("temporal_module.", "temporal_module.net.", 1)
            if (kk not in model_keys) and (cand in model_keys):
                kk = cand
            cand2 = kk.replace("temporal_module.net.", "temporal_module.", 1)
            if (kk not in model_keys) and (cand2 in model_keys):
                kk = cand2

        sd[kk] = v

    # head rename compatibility
    if ("classifier.weight" in model_keys) and ("new_fc.weight" in sd) and ("classifier.weight" not in sd):
        sd["classifier.weight"] = sd["new_fc.weight"]
        sd["classifier.bias"] = sd["new_fc.bias"]
    if ("new_fc.weight" in model_keys) and ("classifier.weight" in sd) and ("new_fc.weight" not in sd):
        sd["new_fc.weight"] = sd["classifier.weight"]
        sd["new_fc.bias"] = sd["classifier.bias"]

    return sd


def load_model(weights_path, test_segments, modality, args):
    """
    Build model consistent with training and load checkpoint.
    Also infer num_class from checkpoint head weight shape.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd_raw = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    # strip module. now (for inference of num_class)
    sd_tmp = {}
    for k, v in sd_raw.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        sd_tmp[kk] = v

    # infer num_class
    if "classifier.weight" in sd_tmp:
        num_class = int(sd_tmp["classifier.weight"].shape[0])
    elif "new_fc.weight" in sd_tmp:
        num_class = int(sd_tmp["new_fc.weight"].shape[0])
    else:
        raise RuntimeError("Cannot infer num_class from checkpoint (no classifier/new_fc).")

    # dataset config: only for root/list/prefix
    _, train_list, val_list, root_path, prefix = dataset_config_1fold.return_dataset(args.dataset, modality)


    arch = "vit"

    net = TSN(
        num_class=num_class,
        num_segments=test_segments,
        modality=modality,
        base_model=arch,
        consensus_type=args.crop_fusion_type,
        dropout=0.0,               # testing: dropout off
        img_feature_dim=args.img_feature_dim,
        partial_bn=False,          # testing: no need
        pretrain=args.pretrain,    # recommend: 'none'
        is_shift=False,
        shift_div=8,
        shift_place="blockres",
        temporal_pool=False,
        non_local=False,
    )

    model_keys = set(net.state_dict().keys())
    sd = _normalize_state_dict_keys(sd_raw, model_keys)

    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return net, num_class, val_list, root_path, prefix, arch


def build_cropping(net, args):
    input_size = net.scale_size if args.full_res else net.input_size

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError(f"Only 1,3,5,10 crops supported, got {args.test_crops}")
    return cropping


def make_loader(root_path, list_file, num_segments, modality, prefix, net, args):
    cropping = build_cropping(net, args)

    new_length = 1 if modality == "RGB" else 5  # must match training


    normalize = GroupNormalize(net.input_mean, net.input_std)

    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
    ])

    dataset = TSNDataSet(
        root_path,
        list_file,
        num_segments=num_segments,
        new_length=new_length,
        modality=modality,
        image_tmpl=prefix,
        test_mode=True,
        remove_missing=True,
        transform=transform,
        dense_sample=args.dense_sample,
        twice_sample=args.twice_sample,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


def forward_one_batch(net, data, num_class, args):
    """
    data: [B, C, H, W] or [B, num_crop*C, H, W]
    reshape -> [B*num_crop, C, H, W] -> forward -> avg crops
    """
    B, C, H, W = data.shape

    num_crop = args.test_crops
    if args.dense_sample:
        num_crop *= 10
    if args.twice_sample:
        num_crop *= 2

    if C % num_crop != 0:
        raise ValueError(
            f"Channel C={C} not divisible by num_crop={num_crop}. "
            f"Try --test_crops 1, or check transform stacking."
        )

    clip_C = C // num_crop
    data_in = data.view(B * num_crop, clip_C, H, W).contiguous()

    rst = net(data_in)  # [B*num_crop, num_class]
    rst = rst.view(B, num_crop, num_class).mean(1)

    if args.softmax:
        rst = F.softmax(rst, dim=1)

    return rst


def main():
    args = build_parser().parse_args()

    weights_list = args.weights.split(',')
    test_segments_list = [int(s) for s in args.test_segments.split(',')]
    assert len(weights_list) == len(test_segments_list), "weights and test_segments length mismatch"

    if args.coeff is None:
        coeff_list = [1.0] * len(weights_list)
    else:
        coeff_list = [float(x) for x in args.coeff.split(',')]
        assert len(coeff_list) == len(weights_list)

    if args.test_list is not None:
        test_file_list = args.test_list.split(',')
        if len(test_file_list) == 1 and len(weights_list) > 1:
            test_file_list = test_file_list * len(weights_list)
        assert len(test_file_list) == len(weights_list)
    else:
        test_file_list = [None] * len(weights_list)

    nets, loaders = [], []
    num_class = None
    total_num = None

    for w, nseg, tfile in zip(weights_list, test_segments_list, test_file_list):
        modality = args.modality

        net, nc, default_val_list, root_path, prefix, arch = load_model(w, nseg, modality, args)
        if num_class is None:
            num_class = nc
        else:
            assert num_class == nc

        list_file = tfile if tfile is not None else default_val_list
        loader = make_loader(root_path, list_file, nseg, modality, prefix, net, args)

        if args.gpus is not None:
            net = torch.nn.DataParallel(net.cuda(), device_ids=args.gpus)
        else:
            net = torch.nn.DataParallel(net.cuda())

        net.eval()
        nets.append(net)
        loaders.append(loader)

        if total_num is None:
            total_num = len(loader.dataset)
        else:
            assert total_num == len(loader.dataset), "ensemble requires identical dataset ordering/length"

    max_num = args.max_num if args.max_num > 0 else total_num

    top1 = AverageMeter()
    topk = AverageMeter()

    outputs = []
    preds_scores = []

    proc_start_time = time.time()

    k_eval = min(5, int(num_class))

    for i, batch_tuple in enumerate(zip(*loaders)):
        if i * args.batch_size >= max_num:
            break

        with torch.no_grad():
            rst_sum = None
            sum_w = 0.0
            this_label = None

            for (data, label), net, coeff in zip(batch_tuple, nets, coeff_list):
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                this_label = label

                rst = forward_one_batch(net, data, num_class, args)  # [B,num_class]
                if rst_sum is None:
                    rst_sum = rst * coeff
                else:
                    rst_sum = rst_sum + rst * coeff
                sum_w += float(coeff)


            ensembled = rst_sum / max(sum_w, 1e-12)  # [B,num_class]
            ensembled_cpu = ensembled.detach().cpu().numpy()

            for p, g in zip(ensembled_cpu, this_label.detach().cpu().numpy()):
                outputs.append([p[None, ...], int(g)])
                preds_scores.append(p)

            prec1, preck = accuracy(ensembled.detach().cpu(), this_label.detach().cpu(), topk=(1, k_eval))
            top1.update(prec1.item(), this_label.numel())
            topk.update(preck.item(), this_label.numel())

            if i % 20 == 0:
                cnt_time = time.time() - proc_start_time
                print(
                    f"batch {i} done, total {i}/{total_num}, avg {cnt_time/(i+1):.3f} sec/batch, "
                    f"moving Prec@1 {top1.avg:.3f} Prec@{k_eval} {topk.avg:.3f}"
                )

    video_pred = [int(np.argmax(x[0])) for x in outputs]
    video_labels = [x[1] for x in outputs]

    cf = confusion_matrix(video_labels, video_pred, labels=list(range(num_class))).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / np.maximum(cls_cnt, 1.0)

    print('-----Evaluation finished------')
    print('Class Accuracy:', cls_acc)
    print('Mean Class Acc {:.02f}%'.format(np.mean(cls_acc) * 100))
    print('Overall Prec@1 {:.02f}% Prec@{} {:.02f}%'.format(top1.avg, k_eval, topk.avg))

    out = {
        "overall_prec1": float(top1.avg),
        f"overall_prec@{k_eval}": float(topk.avg),
        "class_acc": cls_acc.tolist(),
        "confusion_matrix": cf.tolist(),
        "num_class": int(num_class),
        "k_eval": int(k_eval),
    }
    out_path = Path("test_results_summary.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved:", out_path)

    if args.csv_file is not None:

        list_path = (args.test_list.split(',')[0] if args.test_list is not None else loaders[0].dataset.list_file)
        with open(list_path, "r", encoding="utf-8") as f:
            vid_names = [ln.strip().split(' ')[0] for ln in f if ln.strip()]
        assert len(vid_names) == len(video_pred)

        with open(args.csv_file, "w", encoding="utf-8") as f:
            f.write("video,label,pred,score_max\n")
            for n, gt, pr, sc in zip(vid_names, video_labels, video_pred, preds_scores):
                f.write(f"{n},{gt},{pr},{float(np.max(sc)):.6f}\n")
        print("Saved CSV:", args.csv_file)


if __name__ == "__main__":
    main()