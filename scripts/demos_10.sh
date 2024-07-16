#!/bin/bash

# ./scripts/demos_10.sh pick_up_cup yes_no_pick_up.txt yes_no grasped_red_cup_reward cuda:1

# ./scripts/demos_10.sh phone_on_base yes_no_pick_up.txt yes_no phone_grasped_reward cuda:1

# ./scripts/demos_10.sh take_umbrella_out_of_umbrella_stand yes_no_pick_up_umbrella.txt yes_no umbrella_grasped_reward cuda:1

# ./scripts/demos_10.sh put_rubbish_in_bin yes_no_pick_up_rubbish.txt yes_no rubbish_grasped_reward cuda:0
# ./scripts/demos_10.sh put_rubbish_in_bin yes_no_put_in_bin.txt yes_no reward cuda:0

task=$1
prompt=$2
metrics=$3
reward_key=$4
device=$5

python3 scripts/label_demos.py \
    "demos_10/rlbench_${task}/mvmwm_mv_extra_rewards_rot_True/10000/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "new_prompts/${task}/${prompt}" \
    --group="${task}_demos_10" \
    --name="front_10000_${task}_${prompt}_${metrics}" \
    --image-key="front" \
    --label-every=2 \
    --metrics-type ${metrics} \
    --reward-key ${reward_key} \
    --device ${device}

