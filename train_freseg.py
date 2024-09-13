import argparse
import os
import sys

sys.path.append("/data/adhinart/dendrite/scripts/igneous")

from cache_dataloader import CachedDataset
import torch

torch.autograd.set_detect_anomaly(True)
scaler = torch.cuda.amp.GradScaler()
import datetime
import logging
import importlib
import shutil
# import provider
import numpy as np
import wandb

from pathlib import Path
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

classes = [str(i) for i in range(25)]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

    
def ignore_trunk_pc(trunk_id, pc, trunk_pc, label):
    # NOTE: trunk_pc has variable length and cannot be collated using default_collate
    return trunk_id, pc, label

def get_dataloader(path_length: int, num_points: int, fold: int, is_train: bool, batch_size: int, num_workers: int):
    assert fold in [0, 1, 2, 3, 4]


    dataset = CachedDataset(
        f"/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/dataset_-1_{path_length}_{num_points}",
        folds=[
            [3, 5, 11, 12, 23, 28, 29, 32, 39, 42],
            [8, 15, 19, 27, 30, 34, 35, 36, 46, 49],
            [9, 14, 16, 17, 21, 26, 31, 33, 43, 44],
            [2, 6, 7, 13, 18, 24, 25, 38, 41, 50],
            [1, 4, 10, 20, 22, 37, 40, 45, 47, 48],
        ],
        fold=fold,
        is_train=is_train,
        transform=ignore_trunk_pc,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )

    return dataloader



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

def rotation_matrix_3d():
    """
    Generate a random 3D rotation matrix.
    """
    # Random rotation angle for each axis
    angles = np.random.uniform(0, 2*np.pi, 3)
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    
    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def rotate_point_cloud(point_cloud):
    """
    Randomly rotate the point cloud to augment the dataset
    rotation is fully 3D
    Input:
      point_cloud: Nx3 array, original point cloud
    Return:
      rotated_point_cloud: Nx3 array, rotated point cloud
    """
    rotation_matrix = rotation_matrix_3d()
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    return rotated_point_cloud

def rotate_point_cloud_batch(batch_point_cloud):
    """
    Randomly rotate each point cloud in the batch
    rotation is fully 3D
    Input:
      batch_point_cloud: BxNx3 array, original batch of point clouds
    Return:
      rotated_batch_point_cloud: BxNx3 array, rotated batch of point clouds
    """
    rotated_batch_point_cloud = np.zeros(batch_point_cloud.shape, dtype=np.float32)
    for k in range(batch_point_cloud.shape[0]):
        rotated_batch_point_cloud[k, ...] = rotate_point_cloud(batch_point_cloud[k, ...])
    return rotated_batch_point_cloud
    
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
    parser.add_argument("--disable_amp", action="store_true", help="AMP")
    parser.add_argument("--trans", action="store_true", help="whether to use transformation")
    # parser.add_argument("--tnb", action="store_true", help="whether to use transformation")
    parser.add_argument("--ratio", type=float, default=1, help="training data scale")

    parser.add_argument("--path_length", type=int, help="path length")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--num_workers", type=int, default=16, help="num workers")

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    wandb.init(project="freseg_baseline",name = args.log_dir)

    """HYPER PARAMETER"""
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f"freseg_{args.fold}_{args.path_length}_{args.npoint}")
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

    global_epoch = 0
    try:
        best_eval_dice_ann = checkpoint["best_eval_dice_ann"]
        log_string("best_eval_dice_ann:%f" % best_eval_dice_ann)
    except:
        best_eval_dice_ann = 0
        log_string("best_eval_dice_ann:%f" % best_eval_dice_ann)

    # val and test dataloaders are fixed
    testDataLoader = get_dataloader(
        path_length=args.path_length,
        num_points=args.npoint,
        fold=args.fold,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    valDataLoader = testDataLoader

    for epoch in range(start_epoch, args.epoch):
        trainDataLoader = get_dataloader(
            path_length=args.path_length,
            num_points=args.npoint,
            fold=args.fold,
            is_train=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

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
        for i, (trunk_id, points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            # points = points.data.numpy()
            # points = rotate_point_cloud_batch(points[:, :,:3])
            # points = torch.Tensor(points)
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
                # print(seg_pred.shape,target.shape,trans_feat.shape)
                loss = criterion(seg_pred, target, trans_feat)
                wandb.log({"train_loss": loss.item(), "epoch": epoch})
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch >= 0:
            with torch.no_grad():
                eval_dice_ann = []
                eval_dice_ves = []
                eval_acc = []
                
                classifier = classifier.eval()
                for i, (trunk_id, points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                    # points = points.data.numpy()
                    # points = points if args.tnb else points[:, :,:3] 
                    # points = torch.Tensor(points)
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
                        eval_acc.append(correct.item() / (args.batch_size * args.npoint))
                        pred_choice = pred_choice.clone().cpu().numpy()
                        target = target.clone().cpu().numpy()
                        eval_dice_ann.append(2*(target*pred_choice).sum() / (target.sum()+pred_choice.sum()))
                        target = 1 - target
                        pred_choice = 1- pred_choice
                        eval_dice_ves.append(2*(target*pred_choice).sum() / (target.sum()+pred_choice.sum()))

            train_instance_acc = np.mean(mean_correct)
            log_string("Train accuracy is: %.5f" % train_instance_acc)
            eval_acc = np.mean(eval_acc)
            log_string("Test accuracy is: %.5f" % eval_acc)
            eval_dice_ann = np.mean(eval_dice_ann)
            log_string("Test Ann Dice is: %.5f" % eval_dice_ann)
            eval_dice_ves = np.mean(eval_dice_ves)
            log_string("Test Ves Dice is: %.5f" % eval_dice_ves)

            wandb.log({"eval_acc": eval_acc, "epoch": epoch})
            wandb.log({"eval_dice_ann": eval_dice_ann, "epoch": epoch})
            wandb.log({"eval_dice_ves": eval_dice_ves, "epoch": epoch})

            if eval_dice_ann > best_eval_dice_ann:
                logger.info("Save model...")
                best_eval_dice_ann = eval_dice_ann
                savepath = str(checkpoints_dir) + "/best_model.pth"
                log_string("Saving at %s" % savepath)
                state = {
                    "epoch": epoch,
                    "train_acc": train_instance_acc,
                    "acc": eval_acc,
                    "best_dice_ann": best_eval_dice_ann,
                    "eval_dice_ves": eval_dice_ves,
                    "eval_dice_ann": eval_dice_ann,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string("Saving model....")

        global_epoch += 1
    log_string("Training completed.")
    checkpoint = torch.load(savepath)
    log_string("Val Ann Dice is: %.5f" % checkpoint["eval_dice_ann"])
    log_string("Val Ves Dice is: %.5f" % checkpoint["eval_dice_ves"])

    best_model = MODEL.get_model(NUM_CLASSES, normal_channel=False).cuda()
    best_model.load_state_dict(checkpoint["model_state_dict"])
    test_dice_ann, test_dice_ves, test_acc = evaluate(testDataLoader, best_model, criterion, args)
    
    log_string(f"Test Accuracy: {test_acc:.5f}")
    log_string(f"Test Ann Dice: {test_dice_ann:.5f}")
    log_string(f"Test Ves Dice: {test_dice_ves:.5f}")

    
def evaluate(dataloader, model, criterion, args):
    model.eval()
    dice_ann = []
    dice_ves = []
    accuracies = []
    
    with torch.no_grad():
        for points, target, _ in tqdm(dataloader, total=len(dataloader), smoothing=0.9):
            points, target = points.float().cuda(), target.long().cuda()
            label = torch.zeros((points.shape[0], 16)).long().cuda()
            points = points.transpose(2, 1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(not args.disable_amp)):
                seg_pred, _ = model(points, label)
                seg_pred = seg_pred.contiguous().view(-1, 2)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]

                correct = pred_choice.eq(target.data).cpu().sum()
                accuracies.append(correct.item() / (args.batch_size * args.npoint))

                pred_choice = pred_choice.cpu().numpy()
                target = target.cpu().numpy()
                dice_ann.append(2 * (target * pred_choice).sum() / (target.sum() + pred_choice.sum()))
                target = 1 - target
                pred_choice = 1 - pred_choice
                dice_ves.append(2 * (target * pred_choice).sum() / (target.sum() + pred_choice.sum()))

    return np.mean(dice_ann), np.mean(dice_ves), np.mean(accuracies)



if __name__ == "__main__":
    args = parse_args()
    main(args)

