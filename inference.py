import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import importlib

import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from freseg_dataset import FreSegDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="model name")
    parser.add_argument(
        "--model_path", type=str, help="Path to model checkpoint to load"
    )
    parser.add_argument("--folds", nargs="*", type=int, help="folds to use")
    parser.add_argument(
        "--trans", action="store_true", help="whether to use transformation"
    )
    parser.add_argument("--output_dir", type=str, help="Path to save output")
    parser.add_argument("--dataset_path", type=str, default="", metavar="N")
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument("--dry_run", type=int, default=0, metavar="N")
    args = parser.parse_args()

    time_samples = []
    dry_run = args.dry_run
    if dry_run:
        print("Dry run, not saving anything")

    print("using npoint:", args.npoint)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    num_classes = 2
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_classes, normal_channel=False).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    batch_size = 16

    test_dataset = FreSegDataset(
        root=args.dataset_path,
        folds=args.folds,
        npoints=args.npoint,
        eval=True,
        trans=args.trans,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    for fn, ct_list, label_list in tqdm(test_loader):
        if dry_run and (len(time_samples) > dry_run):
            break
        time_samples.append(0)
        all_data = []
        all_segs = []
        all_preds = []

        for batch_idx in tqdm(range(0, len(ct_list), batch_size), leave=False):
            points = ct_list[batch_idx : batch_idx + batch_size]
            points = torch.cat(points, dim=0)
            seg = label_list[batch_idx : batch_idx + batch_size]
            seg = torch.cat(seg, dim=0)

            points, seg = (
                points.to(device),
                seg.to(device),
            )

            label = torch.zeros((points.shape[0], 16)).long().cuda()
            points = points.transpose(2, 1)

            with torch.no_grad():
                # [B, N, 25]
                start = time.time()
                seg_pred, _ = classifier(points, label)
                end = time.time()
                time_samples[-1] += end - start
                seg_pred = seg_pred.reshape(-1, num_classes).contiguous()

            points = points.transpose(2, 1).reshape(-1, 3).contiguous()
            data_np = points.cpu().numpy()
            seg_np = seg.cpu().numpy().reshape(-1)
            pred_np = seg_pred.detach().cpu().numpy()
            all_data.append(data_np)
            all_segs.append(seg_np)
            all_preds.append(pred_np)

        if not dry_run:
            name = fn[0].split("/")[-1].split(".")[0]
            np.savez(
                args.output_dir + "/" + name + ".npz",
                data=np.concatenate(all_data, axis=0),
                seg=np.concatenate(all_segs, axis=0),
                pred=np.concatenate(all_preds, axis=0),
            )
    print(
        f"Average inference time: {np.mean(time_samples):.4f} seconds with std: {np.std(time_samples):.4f}"
    )


if __name__ == "__main__":
    inference()
