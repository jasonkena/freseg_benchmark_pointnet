import sys

sys.path.append("/data/adhinart/dendrite/scripts/igneous")

from pathlib import Path
import os
import argparse
import torch
import importlib

from train_freseg import get_dataloader, evaluate


def get_test_dataloaders(
    path_length: int,
    num_points: int,
    fold: int,
    batch_size: int,
    num_workers: int,
    frenet: bool,
):
    datasets = {}
    datasets["seg_den"] = get_dataloader(
        species="seg_den",
        path_length=path_length,
        num_points=num_points,
        fold=fold,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )
    datasets["mouse"] = get_dataloader(
        species="mouse",
        path_length=path_length,
        num_points=num_points,
        fold=-1,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )
    datasets["human"] = get_dataloader(
        species="human",
        path_length=path_length,
        num_points=num_points,
        fold=-1,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )

    return datasets


def main(args):
    exp_dir = Path("./log/")
    exp_dir = exp_dir.joinpath(
        f"freseg_{args.fold}_{args.path_length}_{args.npoint}_{args.frenet}"
    )
    subdirs = os.listdir(exp_dir)
    assert len(subdirs) == 1, f"multiple timestamps found in {exp_dir}"
    exp_dir = exp_dir.joinpath(subdirs[0])
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    savepath = str(checkpoints_dir) + "/best_model.pth"

    checkpoint = torch.load(savepath)

    MODEL = importlib.import_module(args.model)
    NUM_CLASSES = 2

    best_model = MODEL.get_model(NUM_CLASSES, normal_channel=False).cuda()
    best_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {savepath}")

    dataloaders = get_test_dataloaders(
        args.path_length,
        args.npoint,
        args.fold,
        args.batch_size,
        args.num_workers,
        args.frenet,
    )
    print(f"Loaded dataloaders")

    print(
        f"Loaded model from {savepath}, fold {args.fold}, path_length {args.path_length}, npoint {args.npoint}, frenet {args.frenet}"
    )
    for dataset_name, dataloader in dataloaders.items():
        test_dice_ann, test_dice_ves, test_acc = evaluate(
            dataloader, best_model, None, args
        )

        print(f"Dataset: {dataset_name}")
        print(f"Test Accuracy: {test_acc:.5f}")
        print(f"Test Ann Dice: {test_dice_ann:.5f}")
        print(f"Test Ves Dice: {test_dice_ves:.5f}")


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
    main(args)
