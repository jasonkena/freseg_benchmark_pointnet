default:
    just --list

set export
WANDB__SERVICE_WAIT := "300"


# pathlength 10000 npoints 4096
train cuda fold pathlength npoints:
    CUDA_VISIBLE_DEVICES={{cuda}} python train_freseg.py --model pointnet2_part_seg_msg --fold {{fold}} --path_length {{pathlength}} --npoint {{npoints}} --disable_amp

#CHECKPOINT_BASE_PATH := "/data/adhinart/freseg/Pointnet_Pointnet2_pytorch/log/freseg"

# NOTE change output path
#inference_pointnet2_2048:
#    CUDA_VISIBLE_DEVICES=0 python inference.py --model pointnet2_part_seg_msg --model_path="$CHECKPOINT_BASE_PATH/pointnet2_2048/checkpoints/best_model.pth" --output_dir="$CHECKPOINT_BASE_PATH/pointnet2_2048/outputs" --dataset_path="../freseg_dataset" --npoint=2048 --folds 0 1 2 3 4

#inference_pointnet2_1024:
#    CUDA_VISIBLE_DEVICES=2 python inference.py --model pointnet2_part_seg_msg --model_path="$CHECKPOINT_BASE_PATH/pointnet2_1024/checkpoints/best_model.pth" --output_dir="$CHECKPOINT_BASE_PATH/pointnet2_1024/outputs" --dataset_path="../freseg_dataset" --npoint=1024 --folds 0 1 2 3 4
