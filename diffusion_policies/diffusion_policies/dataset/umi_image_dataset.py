from typing import Dict
import copy

import numpy as np
import torch

from diffusion_policies.common.normalize_util import get_image_identity_normalizer
from diffusion_policies.common.pytorch_util import dict_apply
from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policies.dataset.base_dataset import BaseImageDataset
from diffusion_policies.model_dp_umi.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class UMIImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        zarr_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)

        keys = ["action", *self.rgb_keys, *self.lowdim_keys]
        replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed=seed)

        self.replay_buffer = replay_buffer
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.train_mask = train_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][t_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data
