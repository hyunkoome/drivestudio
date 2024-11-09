#!/bin/bash

# waymo_example_scenes.txt 파일 경로
scene_file="./data/waymo_example_train_scenes.txt"

# 파일을 읽고 주석(#)이 없는 줄에서 씬 번호를 추출
scenes=($(grep -v '^#' "$scene_file" | awk -F',' '{print $1}'))

# 스크립트 인수로 GPU 번호와 WANDB_ENTITY_NAME 받아오기
GPU_NUMBER=${1:-0}  # 첫 번째 인수가 없으면 기본값 0
WANDB_ENTITY_NAME=$2

# 전달받은 값을 출력 (디버깅용)
echo "Using GPU: $GPU_NUMBER"
if [[ -n "$WANDB_ENTITY_NAME" ]]; then
  echo "WANDB_ENTITY_NAME: $WANDB_ENTITY_NAME"
else
  echo "WANDB_ENTITY_NAME is not set. WandB will not be enabled."
fi

# 씬 번호마다 반복 실행
for scene in "${scenes[@]}"; do
  # WandB 관련 옵션 설정
  if [[ -n "$WANDB_ENTITY_NAME" ]]; then
    wandb_args="--enable_wandb --entity ${WANDB_ENTITY_NAME}"
  else
    wandb_args=""
  fi

  export $(cat .env | xargs) && CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python tools/train.py \
      $wandb_args \
      --config_file configs/omnire.yaml \
      --output_root output \
      --project drivestudio \
      --run_name a100_omnire_waymo_3cm_scene_${scene} \
      dataset=waymo/3cams \
      data.scene_idx=${scene} \
      data.start_timestep=0 \
      data.end_timestep=-1
done

# 실행 방법:
# ./train_waymo_3cm_a100_multi_scenes.sh [GPU_NUMBER] [WANDB_ENTITY_NAME]
#
# 예시 1: GPU 번호와 WandB entity를 모두 지정하는 경우
# ./train_waymo_3cm_a100_multi_scenes.sh 0 hyunkookim-me
#
# 예시 2: GPU 번호만 지정하고 WandB entity를 생략하는 경우
# ./train_waymo_3cm_a100_multi_scenes.sh 1
#
# 예시 3: 두 인수를 모두 생략 (기본 GPU 번호는 0, WandB는 비활성화)
# ./train_waymo_3cm_a100_multi_scenes.sh
