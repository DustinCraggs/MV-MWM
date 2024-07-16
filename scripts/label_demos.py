import itertools
import re
import numpy as np
import strictfire
import wandb
import glob

from PIL import Image
from itertools import chain
from functools import partial

from fmrl.labeller import RewardLabeller


def get_image_stream(episodes, key):
    arrays = chain.from_iterable(episode[key] for episode in episodes)
    return list(Image.fromarray(arr) for arr in arrays)


def label_demos(
    demo_dir,
    model_path,
    prompt_path,
    vlm_batch_size=2,
    device="cuda:0",
    label_every=1,
    group="description_labels",
    name="seed_1112",
    wandb_mode="online",
    image_key="front",
    metrics_type=None,
    reward_key="reward",
):
    prompt_text = open(prompt_path).read()
    print(prompt_text)

    prompt_sequence = prompt_text.split("\n")
    print(prompt_sequence)

    config = {
        "prompt_text": prompt_text,
        "demo_dir": demo_dir,
        "model_path": model_path,
        "prompt_path": prompt_path,
        "vlm_batch_size": vlm_batch_size,
        "device": device,
        "label_every": label_every,
        "image_key": image_key,
    }

    wandb_run = wandb.init(
        group=group,
        name=name,
        entity="gyo",
        project="fmrl",
        mode=wandb_mode,
        config=config,
    )

    labeller = RewardLabeller(
        model_path,
        prompt_sequence,
        batch_size=vlm_batch_size,
        device=device,
        label_every=label_every,
        wandb_run=wandb_run,
    )

    # Load episodes:
    episode_files = glob.glob(f"{demo_dir}/*.npz")
    episodes = [np.load(f) for f in episode_files]

    images = get_image_stream(episodes, image_key)
    dones = chain.from_iterable(ep["is_last"] for ep in episodes)

    reward_keys = [k for k in episodes[0].keys() if "reward" in k]
    all_rewards = {
        k: iter(list(chain.from_iterable(ep[k] for ep in episodes)))
        for k in reward_keys
    }

    metrics_generators = []
    if metrics_type == "pick_up_cup_options":
        metrics_generators = get_pick_up_cup_options_metrics_generators()
    elif metrics_type == "phone_on_base_options":
        metrics_generators = get_phone_on_base_options_metrics_generators()
    elif metrics_type == "put_rubbish_in_bin_options":
        metrics_generators = get_put_rubbish_in_bin_options_metrics_generators()
    elif metrics_type == "yes_no":
        metrics_generators = [
            RewardMetricsGenerator(reward_key, get_yes_no_reward),
            # RewardMetricsGenerator(
            #     reward_key,
            #     partial(get_yes_no_reward, use_certainty_label=True),
            #     name="with_certainty",
            # ),
        ]

    logger = ResultLogger(
        wandb_run,
        metrics_generators,
    )

    for img, done in zip(images, dones):
        rewards = {k: next(v) for k, v in all_rewards.items()}
        # if rewards["reward"] == 0:
        #     continue
        labeller.send([rewards], [0], [img], [done])
        results = labeller.receive(max_in_flight_samples=100)
        logger.log_results(results)

    labeller.finish()
    while labeller.num_in_flight_samples > 0:
        results = labeller.receive(max_in_flight_samples=100)
        logger.log_results(results)


class ResultLogger:
    def __init__(self, wandb_run, metrics_generators):
        self._wandb_run = wandb_run
        self._metrics_generators = metrics_generators

    def log_results(self, results):
        for result in results:
            env_rewards, ep_idx, step_idx, img, done, labels, novelty_reward = result
            if labels != None:
                self.log(env_rewards, novelty_reward, labels)

    def log(self, env_rewards, novelty_reward, labels):
        metrics = {"novelty_reward": novelty_reward}
        for metrics_generator in self._metrics_generators:
            metrics.update(metrics_generator.get_metrics(env_rewards, labels))

        self._wandb_run.log(metrics)


class RewardMetricsGenerator:

    def __init__(self, key, reward_func, name=""):
        self._key = key
        self._reward_func = reward_func
        self._name = name
        self._steps = 0
        # Use NaN for divide by zero errors:
        self._tp = np.float64(0)
        self._fp = np.float64(0)
        self._tn = np.float64(0)
        self._fn = np.float64(0)

    def get_metrics(self, env_rewards, labels):
        # label = "\n".join(labels)

        self._steps += 1
        print(f"{env_rewards=}")
        print(f"{self._key=}")
        env_reward = env_rewards[self._key]
        print(f"{env_reward=}")
        vlm_goal_achieved, vlm_reward_logit = self._reward_func(labels)

        self._tp += vlm_goal_achieved and env_reward
        self._fp += vlm_goal_achieved and not env_reward
        self._tn += not vlm_goal_achieved and not env_reward
        self._fn += not vlm_goal_achieved and env_reward

        metrics = {
            "env_reward": int(env_reward),
            "vlm_goal_achieved": int(vlm_goal_achieved),
            "vlm_reward_logit": vlm_reward_logit,
            "true_positive_rate": self._tp / (self._tp + self._fn),
            "false_positive_rate": self._fp / (self._fp + self._tn),
            "true_negative_rate": self._tn / (self._tn + self._fp),
            "false_negative_rate": self._fn / (self._fn + self._tp),
            "accuracy": (self._tp + self._tn) / self._steps,
        }
        name = f"{self._name}/" if len(self._name) else ""
        return {f"{name}{self._key}/{k}": v for k, v in metrics.items()}


def get_yes_no_reward(labels, target="yes", use_certainty_label=False):
    goal_achieved_label = labels[1].lower()
    vlm_goal_achieved = target in goal_achieved_label
    logit = float(vlm_goal_achieved)
    if not vlm_goal_achieved and "no" not in goal_achieved_label:
        logit = 0.5

    if use_certainty_label:
        is_certain = target in labels[2].lower()
        vlm_goal_achieved = vlm_goal_achieved and is_certain
        logit = logit if is_certain else logit - 1

    return vlm_goal_achieved, logit


def get_rank_reward(labels, target_letters, pattern=r"([a-z])\)", achieved_when_top=1):
    # label = "\n".join(labels)
    label = labels[-1]

    # Label should be in format "... a) ... b) ...". Extract the integer after each
    # letter:
    matches = re.findall(pattern, label)
    if len(matches) == 0:
        return 0, 0

    ranked_letters = [m[0] for m in matches]

    for i, letter in enumerate(ranked_letters):
        if letter in target_letters:
            goal_achieved = i < achieved_when_top
            return goal_achieved, 1 - i / len(ranked_letters)
    return 0, 0


def get_pick_up_cup_options_metrics_generators():
    reward_key_to_letters = {
        "reward": ["c"],
        "grasped_red_cup_reward": ["c"],
        "is_proximate_to_red_cup_reward": ["f", "c"],
        "is_proximate_to_other_cup_reward": ["g", "a"],
        "is_far_away_from_both_cups_reward": ["b", "e", "h", "i"],
    }
    return [
        *get_rank_metrics_generators(reward_key_to_letters),
        *get_rank_metrics_generators(reward_key_to_letters, name="top_2", top=2),
    ]


def get_phone_on_base_options_metrics_generators():
    reward_key_to_letters = {
        "reward": ["c"],
        "phone_grasped_reward": ["a"],
        "nothing_grasped_reward": ["e", "h", "i"],
        "phone_on_base_reward": ["c"],
        "phone_on_base_and_nothing_grasped_reward": ["c"],
        "base_grasped_reward": ["b"],
        "is_proximate_to_phone_reward": ["f"],
        "is_proximate_to_base_reward": ["g"],
        "phone_is_proximate_to_base_reward": ["c"],
    }
    return [
        *get_rank_metrics_generators(reward_key_to_letters),
        *get_rank_metrics_generators(reward_key_to_letters, name="top_2", top=2),
    ]


def get_put_rubbish_in_bin_options_metrics_generators():
    reward_key_to_letters = {
        "reward": ["c"],
        "rubbish_grasped_reward": ["a"],
        "tomato_grasped_reward": ["b"],
        "rubbish_in_bin_reward": ["c"],
        "is_proximate_to_rubbish_reward": ["a", "f"],
        "is_proximate_to_bin_reward": ["g"],
        "rubbish_is_proximate_to_bin_reward": ["h", "c"],
        # "is_far_away_from_objects": ["e", "i"],
    }
    return [
        *get_rank_metrics_generators(reward_key_to_letters),
        *get_rank_metrics_generators(reward_key_to_letters, name="top_2", top=2),
    ]


def get_take_umbrella_out_of_stand_options_metrics_generators():
    reward_key_to_letters = {
        "reward": [""],
        "umbrella_grasped_reward": [""],
        "is_proximate_to_umbrella_reward": [""],
        "umbrella_is_proximate_to_target_reward": [""],
    }
    return [
        *get_rank_metrics_generators(reward_key_to_letters),
        *get_rank_metrics_generators(reward_key_to_letters, name="top_2", top=2),
    ]


def get_rank_metrics_generators(reward_key_to_letters, name="", top=1):
    return [
        RewardMetricsGenerator(
            k,
            partial(
                get_rank_reward,
                target_letters=target_letters,
                achieved_when_top=top,
            ),
            name=name,
        )
        for k, target_letters in reward_key_to_letters.items()
    ]


if __name__ == "__main__":
    strictfire.StrictFire(label_demos)
