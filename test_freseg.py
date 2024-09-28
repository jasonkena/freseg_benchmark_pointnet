import sys

sys.path.append("/data/adhinart/dendrite/scripts/igneous")

from pathlib import Path
import os
import argparse
import torch
import importlib
from tqdm import tqdm
import math
import numpy as np
from collections import defaultdict

from train_freseg import get_dataloader
from frenet import get_closest

import glob


def get_test_dataloaders(
    path_length: int,
    fold: int,
    num_workers: int,
    frenet: bool,
):
    num_points = 1000000
    batch_size = 1

    datasets = {}
    files = {}
    datasets["seg_den"], files["seg_den"] = get_dataloader(
        species="seg_den",
        path_length=path_length,
        num_points=num_points,
        fold=fold,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )
    datasets["mouse"], files["mouse"] = get_dataloader(
        species="mouse",
        path_length=path_length,
        num_points=num_points,
        fold=-1,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )
    datasets["human"], files["human"] = get_dataloader(
        species="human",
        path_length=path_length,
        num_points=num_points,
        fold=-1,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )

    return datasets, files


def do_inference(dataloader, model, args, files, output_dir):
    model.eval()
    with torch.no_grad():
        # NOTE: dataloader_idx assumes that shuffle is off (ie that is_train=False)
        for dataloader_idx, (trunk_id, points, target) in tqdm(
            enumerate(dataloader), total=len(dataloader), smoothing=0.9
        ):
            original_points = points
            # NOTE: assumes order already randomized
            # divide points into batches
            assert points.shape[0] == 1 and target.shape[0] == 1
            num_total_points = points.shape[1]

            # ceil
            num_batches = int(math.ceil(num_total_points / args.npoint))
            padding = num_batches * args.npoint - num_total_points

            if padding > 0:
                points = torch.cat([points, points[:, :padding, :]], dim=1)
            assert points.shape[1] % args.npoint == 0

            # now reshape points to (num_batches, npoint, 3) and target to (num_batches, npoint)
            points = points.view(num_batches, args.npoint, 3).float()
            preds = []

            for i in range(math.ceil(num_batches / args.batch_size)):
                points_batch = points[
                    i * args.batch_size : (i + 1) * args.batch_size
                ].cuda()
                label = torch.zeros((points_batch.shape[0], 16)).long().cuda()
                points_batch = points_batch.transpose(2, 1)
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(not args.disable_amp),
                ):
                    seg_pred, _ = model(points_batch, label)
                    seg_pred = seg_pred.contiguous().view(-1, 2)
                    pred_max = seg_pred.max(1)[1]

                    preds.append(pred_max.cpu().numpy())

            pred = np.concatenate(preds, axis=0)[:num_total_points]

            np.savez(
                os.path.join(output_dir, os.path.basename(files[dataloader_idx])),
                trunk_id=trunk_id,
                pc=original_points.squeeze(0).numpy(),
                trunk_pc=np.load(files[dataloader_idx])["trunk_pc"],
                label=target.squeeze(0).numpy(),
                pred=pred,
            )


def inference(args):
    # NOTE: batch_size will be for number of 30k samples out of 1M
    exp_dir = Path("./log/")
    exp_dir = exp_dir.joinpath(
        f"freseg_{args.fold}_{args.path_length}_{args.npoint}_{args.frenet}"
    )
    subdirs = os.listdir(exp_dir)
    assert len(subdirs) == 1, f"multiple timestamps found in {exp_dir}"
    exp_dir = exp_dir.joinpath(subdirs[0])
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    savepath = str(checkpoints_dir) + "/best_model.pth"

    output_path = exp_dir.joinpath("output/")

    checkpoint = torch.load(savepath)

    MODEL = importlib.import_module(args.model)
    NUM_CLASSES = 2

    best_model = MODEL.get_model(NUM_CLASSES, normal_channel=False).cuda()
    best_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {savepath}")

    # args.npoint,
    # args.batch_size,
    dataloaders, files = get_test_dataloaders(
        args.path_length,
        args.fold,
        args.num_workers,
        args.frenet,
    )
    print(f"Loaded dataloaders")

    print(
        f"Loaded model from {savepath}, fold {args.fold}, path_length {args.path_length}, npoint {args.npoint}, frenet {args.frenet}"
    )
    for dataset_name, dataloader in dataloaders.items():
        if not output_path.joinpath(dataset_name).exists():
            output_path.joinpath(dataset_name).mkdir(parents=True)
        do_inference(
            dataloader,
            best_model,
            args,
            files[dataset_name],
            output_path.joinpath(dataset_name),
        )


def metrics(args):
    exp_dir = Path("./log/")
    exp_dir = exp_dir.joinpath(
        f"freseg_{args.fold}_{args.path_length}_{args.npoint}_{args.frenet}"
    )
    subdirs = os.listdir(exp_dir)
    assert len(subdirs) == 1, f"multiple timestamps found in {exp_dir}"
    exp_dir = exp_dir.joinpath(subdirs[0])

    output_path = exp_dir.joinpath("output/")
    datasets = os.listdir(output_path)
    datasets = [d for d in datasets if os.path.isdir(output_path.joinpath(d))]

    results = {}
    for dataset in datasets:
        results[dataset] = defaultdict(int)
        files = os.listdir(output_path.joinpath(dataset))
        for file in tqdm(files):
            data = np.load(output_path.joinpath(dataset).joinpath(file))
            trunk_id = data["trunk_id"].item()
            points = data["pc"]
            pred = data["pred"]
            label = data["label"]

            pred = np.where(label > 0, pred * label, -pred)

            missing_pred = pred == -1
            nonzero_pred = pred > 0

            if np.sum(missing_pred) > 0:
                if np.sum(nonzero_pred) > 0:
                    nonmissing_points = points[nonzero_pred]
                    nonmissing_pred = pred[nonzero_pred]

                    missing_points = points[missing_pred]

                    _, idx = get_closest(missing_points, nonmissing_points)
                    pred[missing_pred] = nonmissing_pred[idx]
                else:
                    print(
                        "false positive spines with no correctly predicted ground truth spines"
                    )

            # create a structured array of pred, label
            structured = np.zeros(len(pred), dtype=[("pred", int), ("label", int)])
            structured["pred"] = pred
            structured["label"] = label

            unique, counts = np.unique(structured, return_counts=True)
            for u, c in zip(unique, counts):
                results[dataset][(trunk_id, u["label"], u["pred"])] += c
        results[dataset] = dict(results[dataset])

    dice = {}
    binary_dice = {}
    for dataset in datasets:
        dice[dataset] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
        binary_dice[dataset] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        for (trunk_id, label, pred), c in results[dataset].items():
            if label == pred:
                dice[dataset][(trunk_id, label)]["tp"] += c
            else:
                dice[dataset][(trunk_id, label)]["fn"] += c
                if pred != -1:
                    dice[dataset][(trunk_id, pred)]["fp"] += c

            if label > 0:
                if pred > 0:
                    binary_dice[dataset][trunk_id]["tp"] += c
                else:
                    binary_dice[dataset][trunk_id]["fn"] += c
            else:
                if pred > 0:
                    binary_dice[dataset][trunk_id]["fp"] += c
                else:
                    binary_dice[dataset][trunk_id]["tn"] += c

        dice[dataset] = dict(dice[dataset])
        binary_dice[dataset] = dict(binary_dice[dataset])
    np.savez(output_path.joinpath("metrics.npz"), dice=dice, binary_dice=binary_dice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--model", type=str, default="pointnet_part_seg", help="model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch Size during training"
    )
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument("--path_length", type=int, help="path length")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--num_workers", type=int, default=16, help="num workers")

    parser.add_argument(
        "--frenet", action="store_true", help="whether to use Frenet transformation"
    )
    parser.add_argument("--disable_amp", action="store_true", help="AMP")

    args = parser.parse_args()

    # inference(args)
    metrics(args)
