#!/bin/bash

export $(cat .env | xargs) && CUDA_VISIBLE_DEVICES=1 python tools/eval_hkkim.py --resume_from output/drivestudio/a100_omnire_waymo_3cm_scene_23/checkpoint_final.pth

#export $(cat .env | xargs) && CUDA_VISIBLE_DEVICES=0 python tools/eval.py --resume_from output/drivestudio/a100_omnire_waymo_3cm_scene_23/checkpoint_final.pth --enable_viewer

# export $(cat .env | xargs) && python tools/train.py --config_file configs/omnire.yaml --output_root output --project drivestudio --run_name omnire_waymo_3cm_scene_23 dataset=waymo/3cams data.scene_idx=23 data.start_timestep=0 data.end_timestep=-1

#    --enable_wandb --entity hyunkookim-me \

#export PYTHONPATH=/home/hyunkoo/DATA/HDD8TB/Add_Objects_DrivingScense/drivestudio:$PYTHONPATH

#    --enable_wandb --entity hyunkookim-me --config_file configs/omnire.yaml --output_root output --project drivestudio --run_name rtx4090_omnire_waymo_3cm_scene_23 dataset=waymo/3cams data.scene_idx=23 data.start_timestep=0 data.end_timestep=-1