default:
    just --list

set export
WANDB__SERVICE_WAIT := "300"


# stability problems with AMP for some reason
train_pointnet2_2048:
    CUDA_VISIBLE_DEVICES=0 python train_freseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_2048 --dataset_path="../freseg_dataset" --folds 1 2 3 4 --disable_amp --npoint=2048

train_pointnet2_2048_trans:
    CUDA_VISIBLE_DEVICES=1 python train_freseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_2048_trans --dataset_path="../freseg_dataset" --folds 1 2 3 4 --disable_amp --npoint=2048 --trans

train_pointnet2_1024:
    CUDA_VISIBLE_DEVICES=2 python train_freseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_1024 --dataset_path="../freseg_dataset" --folds 1 2 3 4 --disable_amp --npoint=1024

train_pointnet2_1024_trans:
    CUDA_VISIBLE_DEVICES=3 python train_freseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_1024_trans --dataset_path="../freseg_dataset" --folds 1 2 3 4 --disable_amp --npoint=1024 --trans

CHECKPOINT_BASE_PATH := "/data/adhinart/freseg/Pointnet_Pointnet2_pytorch/log/freseg"

# NOTE change output path
inference_pointnet2_2048:
    CUDA_VISIBLE_DEVICES=0 python inference.py --model pointnet2_part_seg_msg --model_path="$CHECKPOINT_BASE_PATH/pointnet2_2048/checkpoints/best_model.pth" --output_dir="$CHECKPOINT_BASE_PATH/pointnet2_2048/outputs" --dataset_path="../freseg_dataset" --npoint=2048 --folds 0 1 2 3 4

inference_pointnet2_2048_trans:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet2_part_seg_msg --model_path="$CHECKPOINT_BASE_PATH/pointnet2_2048_trans/checkpoints/best_model.pth" --output_dir="$CHECKPOINT_BASE_PATH/pointnet2_2048_trans/outputs" --dataset_path="../freseg_dataset" --npoint=2048 --trans --folds 0 1 2 3 4

inference_pointnet2_1024:
    CUDA_VISIBLE_DEVICES=2 python inference.py --model pointnet2_part_seg_msg --model_path="$CHECKPOINT_BASE_PATH/pointnet2_1024/checkpoints/best_model.pth" --output_dir="$CHECKPOINT_BASE_PATH/pointnet2_1024/outputs" --dataset_path="../freseg_dataset" --npoint=1024 --folds 0 1 2 3 4

inference_pointnet2_1024_trans:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="$CHECKPOINT_BASE_PATH/pointnet2_1024_trans/checkpoints/best_model.pth" --output_dir="$CHECKPOINT_BASE_PATH/pointnet2_1024_trans/outputs" --dataset_path="../freseg_dataset" --npoint=1024 --trans --folds 0 1 2 3 4
