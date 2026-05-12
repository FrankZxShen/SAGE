# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.nav_dataset import NavDataset
from .config import DataConfig


def _parse_mixture_spec(spec: str) -> tuple[list[str], Optional[list[float]]]:
    if "," in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    elif ";" in spec:
        parts = [p.strip() for p in spec.split(";") if p.strip()]
    else:
        return [spec.strip()], None

    paths: list[str] = []
    weights: list[float] = []
    has_any_weight = False

    for part in parts:
        if ":" in part:
            path, weight_str = part.rsplit(":", 1)
            paths.append(path.strip())
            weights.append(float(weight_str))
            has_any_weight = True
        else:
            paths.append(part)
            weights.append(1.0)

    if not has_any_weight:
        return paths, None
    return paths, weights


def _build_dataset(
    config: DataConfig,
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    split: str,
) -> torch.utils.data.Dataset:
    if "CG-DATA" in data_path:
        return NavDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            max_prompt_length=config.max_prompt_length,
            split=split,
            val_ratio=config.val_ratio,
            rollout_ratio=config.rollout_ratio,
            usage_ratio=config.usage_ratio,
            use_three_images=config.use_three_images,
        )

    return RLHFDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )


def _make_per_sample_weights(datasets: list[torch.utils.data.Dataset], weights: list[float]) -> torch.DoubleTensor:
    if len(datasets) != len(weights):
        raise ValueError("Mixture weights must match the number of datasets.")

    lengths = [len(d) for d in datasets]
    if any(l <= 0 for l in lengths):
        parts: list[str] = []
        for idx, (dataset, length, weight) in enumerate(zip(datasets, lengths, weights)):
            data_path = getattr(dataset, "data_path", None)
            parts.append(
                f"[{idx}] {dataset.__class__.__name__}(data_path={data_path!r}) len={length} weight={weight}"
            )
        raise ValueError("All datasets in mixture must be non-empty:\n" + "\n".join(parts))

    per_sample_weights = torch.empty(sum(lengths), dtype=torch.double)
    offset = 0
    for dataset_weight, dataset_len in zip(weights, lengths):
        per_sample_weights[offset : offset + dataset_len] = float(dataset_weight) / float(dataset_len)
        offset += dataset_len
    return per_sample_weights


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_paths, train_weights = _parse_mixture_spec(config.train_files)
    if len(train_paths) == 1:
        train_dataset = _build_dataset(config, train_paths[0], tokenizer, processor, split="train")
        train_datasets = None
    else:
        train_datasets = [_build_dataset(config, p, tokenizer, processor, split="train") for p in train_paths]
        train_dataset = ConcatDataset(train_datasets)

    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        if train_weights is not None:
            if train_datasets is None:
                raise ValueError("Mixture weights require multiple train datasets.")
            per_sample_weights = _make_per_sample_weights(train_datasets, train_weights)
            sampler = WeightedRandomSampler(
                weights=per_sample_weights,
                num_samples=len(per_sample_weights),
                replacement=True,
                generator=train_dataloader_generator,
            )
        else:
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_paths, _ = _parse_mixture_spec(config.val_files)
    if len(val_paths) == 1:
        val_dataset = _build_dataset(config, val_paths[0], tokenizer, processor, split="val")
    else:
        val_datasets = [_build_dataset(config, p, tokenizer, processor, split="val") for p in val_paths]
        val_dataset = ConcatDataset(val_datasets)

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
