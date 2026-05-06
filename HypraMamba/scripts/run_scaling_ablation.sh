CUDA_VISIBLE_DEVICES=1 python train.py --dataset_index 5 --ablation c3_add --outer_residual_mode scaled --outer_residual_alpha 0.5 > logs/salinas_c3_add_scaled.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_index 4 --ablation full --outer_residual_mode scaled --outer_residual_alpha 0.5 > logs/longkou_full_scaled.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_index 4 --ablation c3_add --outer_residual_mode scaled --outer_residual_alpha 0.5 > logs/longkou_c3_add_scaled.log 2>&1 &
