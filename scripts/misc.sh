python3 scripts/label_demos.py \
    "demos/rlbench_pick_up_cup/mvmwm_mv_rotTrue/1113/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "" \
    --group="simple_label_testing" \
    --name="wrist_seed_summarise_the_state_1113"

python3 scripts/label_demos.py \
    "demos/rlbench_pick_up_cup/mvmwm_mv_rotTrue/1113/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "This is a view from the perspective of the gripper of a robotic arm operating on a table. Summarise the state succinctly in a way that is relevant to the robot." \
    --group="simple_label_testing" \
    --name="wrist_seed_summarise_the_state_1113"

python3 scripts/label_demos.py \
    "demos/rlbench_pick_up_cup/mvmwm_mv_rotTrue/1112/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    'This is a view from the perspective of the gripper of a robotic arm operating on a table. List the goals that the robot has achieved in the image in the format "a) ... b) ...". If no distinct goals have been achieved, reply "none".' \
    --group="simple_label_testing" \
    --name="wrist_seed_summarise_the_state_1112"

python3 scripts/label_demos.py \
    "demos/rlbench_pick_up_cup/mvmwm_mv_rotTrue/1112/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "prompts/options_1.txt" \
    --group="simple_label_testing" \
    --name="wrist_seed_summarise_the_state_1112"


python3 scripts/label_demos.py \
    "demos/rlbench_pick_up_cup/mvmwm_mv_rotTrue/1113/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "prompts/options_1.txt" \
    --group="reward_testing" \
    --name="front_1113_rank_options" --use-yes-no-reward --image-key="front"


python3 scripts/label_demos.py \
    "demos/rlbench_pick_up_cup/mvmwm_mv_extra_rewards_rot_True/1116/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "prompts/rank_options_2.txt" \
    --group="reward_testing" \
    --name="front_1113_rank_options" --use-yes-no-reward --image-key="front"


python3 scripts/label_demos.py \
    "demos/rlbench_phone_on_base/mvmwm_mv_extra_rewards_rot_True/1112/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "prompts/phone_on_base/rank_options_1.txt" \
    --group="phone_on_base_reward_testing" \
    --name="front_1112_rank_options" --image-key="front" --label-every=2 --metrics-type phone_on_base_options

python3 scripts/label_demos.py \
    "demos/rlbench_phone_on_base/mvmwm_mv_extra_rewards_rot_True/1112/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "prompts/phone_on_base/yes_no_pick_up_phone.txt" \
    --group="phone_on_base_reward_testing" \
    --name="front_1112_yes_no_pick_up_phone" --image-key="front" --label-every=2 --metrics-type="yes_no" --reward-key="phone_grasped_reward"

python3 scripts/label_demos.py \
    "demos/rlbench_put_rubbish_in_bin/mvmwm_mv_extra_rewards_rot_True/1111/train_episodes/" \
    "/models/cogvlm2-llama3-chat-19B-int4" \
    "prompts/put_rubbish_in_bin/rank_options_1.txt" \
    --group="put_rubbish_in_bin_reward_testing" \
    --name="front_1111_rank_options_put_rubbish_in_bin" --image-key="front" \
    --label-every=2 --metrics-type put_rubbish_in_bin_options