import glob
import numpy as np
from collections import defaultdict


def aggregate_metrics():
    RECALL_THRESHOLD = 0.7

    files = sorted(
        glob.glob(
            "/data/adhinart/freseg/Pointnet_Pointnet2_pytorch/log/*/*/output/metrics.npz"
        )
    )
    for file in files:
        print(file)
        metrics = np.load(file, allow_pickle=True)["dice"].item()
        binary_metrics = np.load(file, allow_pickle=True)["binary_dice"].item()

        for dataset in metrics.keys():
            new_metrics = defaultdict(dict)
            for (trunk_id, label), values in metrics[dataset].items():
                new_metrics[trunk_id][label] = values

            agg = defaultdict(list)
            for trunk_id in new_metrics.keys():
                d = new_metrics[trunk_id]
                bd = binary_metrics[dataset][trunk_id]

                trunk_dice = 2 * d[0]["tp"] / (2 * d[0]["tp"] + d[0]["fn"] + d[0]["fp"])
                trunk_iou = d[0]["tp"] / (d[0]["tp"] + d[0]["fn"] + d[0]["fp"])

                binary_trunk_dice = 2 * bd["tn"] / (2 * bd["tn"] + bd["fn"] + bd["fp"])
                binary_trunk_iou = bd["tn"] / (bd["tn"] + bd["fn"] + bd["fp"])
                binary_spine_dice = 2 * bd["tp"] / (2 * bd["tp"] + bd["fn"] + bd["fp"])
                binary_spine_iou = bd["tp"] / (bd["tp"] + bd["fn"] + bd["fp"])

                spine_dice = [
                    2
                    * d[label]["tp"]
                    / (2 * d[label]["tp"] + d[label]["fn"] + d[label]["fp"])
                    for label in d.keys()
                    if label > 0
                ]
                spine_iou = [
                    d[label]["tp"] / (d[label]["tp"] + d[label]["fn"] + d[label]["fp"])
                    for label in d.keys()
                    if label > 0
                ]
                spine_dice, spine_iou = np.mean(spine_dice), np.mean(spine_iou)

                spine_recall = [
                    d[label]["tp"] / (d[label]["tp"] + d[label]["fn"])
                    for label in d.keys()
                    if label > 0
                ]
                spine_recall = [r >= RECALL_THRESHOLD for r in spine_recall]
                spine_recall = np.mean(spine_recall)

                agg["trunk_dice"].append(trunk_dice)
                agg["trunk_iou"].append(trunk_iou)
                agg["binary_trunk_dice"].append(binary_trunk_dice)
                agg["binary_trunk_iou"].append(binary_trunk_iou)
                agg["binary_spine_dice"].append(binary_spine_dice)
                agg["binary_spine_iou"].append(binary_spine_iou)
                agg["spine_dice"].append(spine_dice)
                agg["spine_iou"].append(spine_iou)
                agg[f"{RECALL_THRESHOLD}_spine_recall"].append(spine_recall)

            agg = {k: np.mean(v) for k, v in agg.items()}
            print(dataset, agg)
        print()


if __name__ == "__main__":
    aggregate_metrics()
