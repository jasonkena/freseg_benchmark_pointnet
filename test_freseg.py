import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from pathlib import Path
import os
import argparse
import torch
import importlib

from freseg_inference import evaluation
from functools import partial


def load_model(output_dir, model_name):
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    savepath = os.path.join(checkpoints_dir, "best_model.pth")

    checkpoint = torch.load(savepath)

    MODEL = importlib.import_module(model_name)
    NUM_CLASSES = 2

    best_model = MODEL.get_model(NUM_CLASSES, normal_channel=False).cuda()
    best_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {savepath}")

    return best_model


def model_inference(model, points, disable_amp):
    # take in model, [B, N, 3], output [B * N] binary predictions in torch
    label = torch.zeros((points.shape[0], 16)).long().cuda()
    points = points.transpose(2, 1)
    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(not disable_amp),
    ):
        seg_pred, _ = model(points, label)
        seg_pred = seg_pred.contiguous().view(-1, 2)
        pred_max = seg_pred.max(1)[1]

    return pred_max


def main(args):
    # NOTE: batch_size will be for number of 30k samples out of 1M
    exp_dir = Path("./log/")
    exp_dir = exp_dir.joinpath(
        f"freseg_{args.fold}_{args.path_length}_{args.npoint}_{args.frenet}"
    )
    subdirs = os.listdir(exp_dir)
    assert len(subdirs) == 1, f"multiple timestamps found in {exp_dir}"
    exp_dir = str(exp_dir.joinpath(subdirs[0]))

    evaluation(
        output_path=exp_dir,
        fold=args.fold,
        path_length=args.path_length,
        npoint=args.npoint,
        frenet=args.frenet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_model=partial(load_model, model_name=args.model),
        model_inference=partial(model_inference, disable_amp=args.disable_amp),
    )


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
