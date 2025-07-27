import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from ...protocol import Processable
from ...utils.train_utils import TrainUtilities


@dataclass
class VisionCollator:
    processor: Processable

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.processor.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = self.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        # position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).repeat(
        # input_ids.shape[0], 1
        # )
        # position_ids[~attention_mask] = 0
        batched_inputs = {}
        for key, values in inputs.items():
            batched_inputs[key] = torch.concatenate(values, dim=0)
        batched_inputs["input_ids"] = input_ids
        batched_inputs["labels"] = labels
        batched_inputs["attention_mask"] = attention_mask
        # batched_inputs["position_ids"] = position_ids

        return batched_inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
