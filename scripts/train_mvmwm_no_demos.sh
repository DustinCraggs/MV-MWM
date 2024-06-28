TASK=$1
USE_ROTATION=$2
GPU=$3
SEED=$4

DISPLAY=:99 TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python3 mvmwm/train.py \
    --logdir ./logs/${TASK}/mvmwm_mv_rot${USE_ROTATION}/${SEED} \
    --camera_keys 'front|wrist' \
    --control_input 'front|wrist' \
    --task ${TASK} \
    --prefill 20000 \
    --mae.view_masking 1 \
    --mae.viewpoint_pos_emb True \
    --steps 810000 \
    --num_demos 0 \
    --use_rotation ${USE_ROTATION} \
    --seed ${SEED} \
    --demo_bc False \
    --mae_pretrain 10000 \
    --pretrain 1000 \
    --shaped_rewards False \
    --wandb.name no_shaping_seed_${SEED} \
    --wandb.group random_rollout_pretrain_no_shaping
    # --shaped_rewards True \
    # --wandb.name prox_shaped_seed_${SEED} \
    # --wandb.group random_rollout_pretrain_prox_shaped
