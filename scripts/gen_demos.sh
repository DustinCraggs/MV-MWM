# source scripts/gen_demos.sh rlbench_phone_on_base True 0 1111 1024 128
TASK=$1
USE_ROTATION=$2
GPU=$3
SEED=$4
IMG_SIZE=$5
PATCH_SIZE=$6

# source scripts/gen_demos.sh rlbench_phone_on_base True 0 10000 1024 128

DISPLAY=:99 TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python3 mvmwm/train.py \
    --logdir ./demos_10/${TASK}/mvmwm_mv_extra_rewards_rot_${USE_ROTATION}/${SEED} \
    --save_replay True \
    --vlm_rewards.enabled True \
    --vlm_rewards.label_every 250 \
    --render_size "$IMG_SIZE, $IMG_SIZE" \
    --mae.img_h_size $IMG_SIZE \
    --mae.img_w_size $IMG_SIZE \
    --mae.patch_size $PATCH_SIZE \
    --camera_keys 'front|wrist' \
    --control_input 'front|wrist' \
    --task ${TASK} \
    --prefill 0 \
    --mae.view_masking 1 \
    --mae.viewpoint_pos_emb True \
    --steps 810000 \
    --num_demos 10 \
    --use_rotation ${USE_ROTATION} \
    --seed ${SEED} \
    --demo_bc False \
    --mae_pretrain 10000 \
    --pretrain 1000 \
    --shaped_rewards False \
    --wandb.name no_shaping_seed_${SEED} \
    --wandb.group random_rollout_pretrain_no_shaping
    # --prefill 20000 \
    # --shaped_rewards True \
    # --wandb.name prox_shaped_seed_${SEED} \
    # --wandb.group random_rollout_pretrain_prox_shaped
