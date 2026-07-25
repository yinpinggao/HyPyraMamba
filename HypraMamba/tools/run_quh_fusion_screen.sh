#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/gyp/HyPyraMamba/HypraMamba}
cd "$REPO_DIR"
mkdir -p logs

echo "[info] validation-only screen; test set evaluation is disabled"
echo "[info] seeds=0,1,2,3,4"
echo "[info] gpus=3,4,5"

echo "[dataset] qingyun"
pids=()
echo "[launch] dataset=qingyun scale=1.0 gpu=3 exp=RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE1"
CUDA_VISIBLE_DEVICES=3 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 11 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE1 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric oa --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 1.0 --class_weight_multipliers 6:1.10 > logs/quh_valscreen_qingyun_spe_scale1.log 2>&1 < /dev/null &
pids+=("$!")
echo "[launch] dataset=qingyun scale=0.75 gpu=4 exp=RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE0P75"
CUDA_VISIBLE_DEVICES=4 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 11 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE0P75 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric oa --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 0.75 --class_weight_multipliers 6:1.10 > logs/quh_valscreen_qingyun_spe_scale0p75.log 2>&1 < /dev/null &
pids+=("$!")
echo "[launch] dataset=qingyun scale=0.5 gpu=5 exp=RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE0P5"
CUDA_VISIBLE_DEVICES=5 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 11 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE0P5 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric oa --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 0.5 --class_weight_multipliers 6:1.10 > logs/quh_valscreen_qingyun_spe_scale0p5.log 2>&1 < /dev/null &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
  echo "[failed] at least one validation-screen job failed" >&2
  exit "$status"
fi
pids=()
echo "[launch] dataset=qingyun scale=0.25 gpu=3 exp=RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE0P25"
CUDA_VISIBLE_DEVICES=3 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 11 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_QINGYUN_SPE_SCALE0P25 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric oa --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 0.25 --class_weight_multipliers 6:1.10 > logs/quh_valscreen_qingyun_spe_scale0p25.log 2>&1 < /dev/null &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
  echo "[failed] at least one validation-screen job failed" >&2
  exit "$status"
fi

echo "[dataset] tangdaowan"
pids=()
echo "[launch] dataset=tangdaowan scale=1.0 gpu=3 exp=RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE1"
CUDA_VISIBLE_DEVICES=3 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 12 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE1 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric miou --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 1.0 > logs/quh_valscreen_tangdaowan_spe_scale1.log 2>&1 < /dev/null &
pids+=("$!")
echo "[launch] dataset=tangdaowan scale=0.75 gpu=4 exp=RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE0P75"
CUDA_VISIBLE_DEVICES=4 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 12 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE0P75 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric miou --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 0.75 > logs/quh_valscreen_tangdaowan_spe_scale0p75.log 2>&1 < /dev/null &
pids+=("$!")
echo "[launch] dataset=tangdaowan scale=0.5 gpu=5 exp=RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE0P5"
CUDA_VISIBLE_DEVICES=5 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 12 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE0P5 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric miou --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 0.5 > logs/quh_valscreen_tangdaowan_spe_scale0p5.log 2>&1 < /dev/null &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
  echo "[failed] at least one validation-screen job failed" >&2
  exit "$status"
fi
pids=()
echo "[launch] dataset=tangdaowan scale=0.25 gpu=3 exp=RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE0P25"
CUDA_VISIBLE_DEVICES=3 nohup /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -u train.py --dataset_index 12 --data_set_path ./data --split_dir ./splits/quh_100_30_seed0-9 --exp_name RUNS_QUH_VALSCREEN_TANGDAOWAN_SPE_SCALE0P25 --train_samples 100 --val_samples 30 --seed_list 0,1,2,3,4 --max_epoch 200 --tile_size 512 --tile_overlap 32 --tile_update_groups 2 --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 --label_smoothing 0.05 --class_weight_mode balanced --checkpoint_metric miou --checkpoint_tie_break secondary --evaluate_test false --gaussian_sigma 1.0 --gaussian_spectral_sigma 1.0 --spectral_diff_alpha 0.5 --spectral_fusion_scale 0.25 > logs/quh_valscreen_tangdaowan_spe_scale0p25.log 2>&1 < /dev/null &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
  echo "[failed] at least one validation-screen job failed" >&2
  exit "$status"
fi

echo "[done] validation-only fusion screen completed"
echo "[next] compare mean_validation_result.txt files; do not inspect test metrics"
