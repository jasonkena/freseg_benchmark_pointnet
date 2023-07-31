default:
    just --list

time_num_samples := "10"
# networks default to 30k input points

train_pointnet1:
    CUDA_VISIBLE_DEVICES=0 python train_ribseg.py --model pointnet_part_seg --log_dir pointnet1 --dataset_path="../ribseg_benchmark"

train_pointnet1_binary:
    CUDA_VISIBLE_DEVICES=1 python train_ribseg.py --model pointnet_part_seg --log_dir pointnet1_binary --binary --dataset_path="../ribseg_benchmark"

# note that second stage doesn't run with binary flag since it predicts over 25 classes
train_pointnet1_binary_second_stage:
    CUDA_VISIBLE_DEVICES=1 python train_ribseg.py --model pointnet_part_seg --log_dir pointnet1_binary_second_stage --dataset_path="/data/adhinart/ribseg/outputs/pointnet1" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet1_binary"

inference_pointnet1:
    CUDA_VISIBLE_DEVICES=0 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1/outputs" --dataset_path="../ribseg_benchmark"

inference_pointnet1_binary:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary/outputs" --dataset_path="../ribseg_benchmark" --binary

inference_pointnet1_binary_second_stage:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet1" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet1_binary"

time_pointnet1:
    CUDA_VISIBLE_DEVICES=0 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1/outputs" --dataset_path="../ribseg_benchmark" --dry_run={{time_num_samples}}
time_pointnet1_binary:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary/outputs" --dataset_path="../ribseg_benchmark" --binary --dry_run={{time_num_samples}}

time_pointnet1_binary_second_stage:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet1" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet1_binary" --dry_run={{time_num_samples}}

# stability problems with AMP for some reason
train_pointnet2:
    CUDA_VISIBLE_DEVICES=2 python train_ribseg.py --model pointnet2_part_seg_msg --log_dir pointnet2 --dataset_path="../ribseg_benchmark" --disable_amp

train_pointnet2_binary:
    CUDA_VISIBLE_DEVICES=3 python train_ribseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_binary --binary --dataset_path="../ribseg_benchmark" --disable_amp

# note that second stage doesn't run with binary flag since it predicts over 25 classes
train_pointnet2_binary_second_stage:
    CUDA_VISIBLE_DEVICES=3 python train_ribseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_binary_second_stage --dataset_path="/data/adhinart/ribseg/outputs/pointnet2" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet2_binary" --disable_amp

inference_pointnet2:
    CUDA_VISIBLE_DEVICES=2 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2/outputs" --dataset_path="../ribseg_benchmark"

inference_pointnet2_binary:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary/outputs" --dataset_path="../ribseg_benchmark" --binary

inference_pointnet2_binary_second_stage:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet2" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet2_binary"

time_pointnet2:
    CUDA_VISIBLE_DEVICES=2 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2/outputs" --dataset_path="../ribseg_benchmark" --dry_run={{time_num_samples}}

time_pointnet2_binary:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary/outputs" --dataset_path="../ribseg_benchmark" --binary --dry_run={{time_num_samples}}

time_pointnet2_binary_second_stage:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet2" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet2_binary" --dry_run={{time_num_samples}}


### 2048 points
train_pointnet1_2048:
    CUDA_VISIBLE_DEVICES=0 python train_ribseg.py --model pointnet_part_seg --log_dir pointnet1_2048 --dataset_path="../ribseg_benchmark" --npoint=2048

train_pointnet1_2048_binary:
    CUDA_VISIBLE_DEVICES=1 python train_ribseg.py --model pointnet_part_seg --log_dir pointnet1_2048_binary --binary --dataset_path="../ribseg_benchmark" --npoint=2048

# note that second stage doesn't run with binary flag since it predicts over 25 classes
train_pointnet1_2048_binary_second_stage:
    CUDA_VISIBLE_DEVICES=1 python train_ribseg.py --model pointnet_part_seg --log_dir pointnet1_2048_binary_second_stage --dataset_path="/data/adhinart/ribseg/outputs/pointnet1_2048" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet1_2048_binary" --npoint=2048

inference_pointnet1_2048:
    CUDA_VISIBLE_DEVICES=0 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048/outputs" --dataset_path="../ribseg_benchmark" --npoint=2048

inference_pointnet1_2048_binary:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary/outputs" --dataset_path="../ribseg_benchmark" --binary --npoint=2048

inference_pointnet1_2048_binary_second_stage:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet1_2048" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet1_2048_binary" --npoint=2048

time_pointnet1_2048:
    CUDA_VISIBLE_DEVICES=0 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048/outputs" --dataset_path="../ribseg_benchmark" --npoint=2048 --dry_run={{time_num_samples}}

time_pointnet1_2048_binary:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary/outputs" --dataset_path="../ribseg_benchmark" --binary --npoint=2048 --dry_run={{time_num_samples}}

time_pointnet1_2048_binary_second_stage:
    CUDA_VISIBLE_DEVICES=1 python inference.py --model pointnet_part_seg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet1_2048" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet1_2048_binary" --npoint=2048 --dry_run={{time_num_samples}}

# stability problems with AMP for some reason
train_pointnet2_2048:
    CUDA_VISIBLE_DEVICES=2 python train_ribseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_2048 --dataset_path="../ribseg_benchmark" --disable_amp --npoint=2048

train_pointnet2_2048_binary:
    CUDA_VISIBLE_DEVICES=3 python train_ribseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_2048_binary --binary --dataset_path="../ribseg_benchmark" --disable_amp --npoint=2048

# note that second stage doesn't run with binary flag since it predicts over 25 classes
train_pointnet2_2048_binary_second_stage:
    CUDA_VISIBLE_DEVICES=3 python train_ribseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_2048_binary_second_stage --dataset_path="/data/adhinart/ribseg/outputs/pointnet2_2048" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet2_2048_binary" --disable_amp --npoint=2048

inference_pointnet2_2048:
    CUDA_VISIBLE_DEVICES=2 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048/outputs" --dataset_path="../ribseg_benchmark" --npoint=2048

inference_pointnet2_2048_binary:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary/outputs" --dataset_path="../ribseg_benchmark" --binary --npoint=2048

inference_pointnet2_2048_binary_second_stage:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet2_2048" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet2_2048_binary" --npoint=2048

time_pointnet2_2048:
    CUDA_VISIBLE_DEVICES=2 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048/outputs" --dataset_path="../ribseg_benchmark" --npoint=2048 --dry_run={{time_num_samples}}

time_pointnet2_2048_binary:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary/outputs" --dataset_path="../ribseg_benchmark" --binary --npoint=2048 --dry_run={{time_num_samples}}

time_pointnet2_2048_binary_second_stage:
    CUDA_VISIBLE_DEVICES=3 python inference.py --model pointnet2_part_seg_msg --model_path="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary_second_stage/checkpoints/best_model.pth" --output_dir="/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary_second_stage/outputs" --dataset_path="/data/adhinart/ribseg/outputs/pointnet2_2048" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointnet2_2048_binary" --npoint=2048 --dry_run={{time_num_samples}}
