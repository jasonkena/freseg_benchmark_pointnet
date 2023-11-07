"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from freseg_dataset import FreSegDataset

import torch

torch.autograd.set_detect_anomaly(True)
scaler = torch.cuda.amp.GradScaler()
import datetime
import logging
import importlib
import shutil
import provider
import numpy as np
import wandb

wandb.init(project="freseg")

from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

classes = [str(i) for i in range(25)]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser("Model")
    parser.add_argument(
        "--model", type=str, default="pointnet_part_seg", help="model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch Size during training"
    )
    parser.add_argument("--epoch", default=200, type=int, help="epoch to run")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="initial learning rate"
    )
    # parser.add_argument("--gpu", type=str, default="0", help="specify GPU devices")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Adam or SGD")
    parser.add_argument("--log_dir", type=str, default=None, help="log path")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    # parser.add_argument("--npoint", type=int, default=2048, help="point Number")
    parser.add_argument(
        "--step_size", type=int, default=20, help="decay step for lr decay"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="decay rate for lr decay"
    )
    parser.add_argument("--dataset_path", type=str, default="", metavar="N")
    parser.add_argument("--disable_amp", action="store_true", help="AMP")
    parser.add_argument("--folds", nargs="*", type=int, help="folds to use")
    parser.add_argument("--test_folds", nargs="*", type=int, help="folds to use")
    parser.add_argument(
        "--trans", action="store_true", help="whether to use transformation"
    )

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """HYPER PARAMETER"""
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath("freseg")
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    TRAIN_DATASET = FreSegDataset(
        root=args.dataset_path, npoints=args.npoint, folds=args.folds, trans=args.trans
    )
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    # TEST_DATASET = FreSegDataset(
    #     root=args.dataset_path,
    #     npoints=args.npoint,
    #     folds=args.test_folds,
    #     trans=args.trans,
    # )
    # testDataLoader = torch.utils.data.DataLoader(
    #     TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10
    # )
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    # log_string("The number of test data is: %d" % len(TEST_DATASET))

    NUM_CLASSES = 2

    """MODEL LOADING"""
    MODEL = importlib.import_module(args.model)
    shutil.copy("models/%s.py" % args.model, str(exp_dir))
    shutil.copy("models/pointnet2_utils.py", str(exp_dir))

    classifier = MODEL.get_model(NUM_CLASSES, normal_channel=False).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + "/checkpoints/best_model.pth")
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.learning_rate, momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))
        """Adjust learning rate and BN momentum"""
        lr = max(
            args.learning_rate * (args.lr_decay ** (epoch // args.step_size)),
            LEARNING_RATE_CLIP,
        )
        log_string("Learning rate:%f" % lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        momentum = MOMENTUM_ORIGINAL * (
            MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)
        )
        if momentum < 0.01:
            momentum = 0.01
        print("BN momentum updated to: %f" % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        """learning one epoch"""
        for i, (points, target) in tqdm(
            enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()

            points = points.data.numpy()
            # disable scale and shift augmentations
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = (
                points.float().cuda(),
                target.long().cuda(),
            )
            label = torch.zeros((points.shape[0], 16)).long().cuda()
            points = points.transpose(2, 1)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(not args.disable_amp)
            ):
                seg_pred, trans_feat = classifier(points, label)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]

                correct = pred_choice.eq(target.data).cpu().sum()
                mean_correct.append(correct.item() / (args.batch_size * args.npoint))
                loss = criterion(seg_pred, target, trans_feat)
                wandb.log({"train_loss": loss.item(), "epoch": epoch})
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_instance_acc = np.mean(mean_correct)
        log_string("Train accuracy is: %.5f" % train_instance_acc)

        logger.info("Save model...")
        savepath = str(checkpoints_dir) + "/best_model.pth"
        log_string("Saving at %s" % savepath)
        state = {
            "epoch": epoch,
            "train_acc": train_instance_acc,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, savepath)
        log_string("Saving model....")

        log_string("Best accuracy is: %.5f" % best_acc)
        log_string("Best class avg mIOU is: %.5f" % best_class_avg_iou)
        log_string("Best inctance avg mIOU is: %.5f" % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
