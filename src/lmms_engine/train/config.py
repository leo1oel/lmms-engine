from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Union

import transformers

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    only_save_mm_adapter: Optional[bool] = False
    use_rmpad: Optional[bool] = False
    fsdp2: Optional[bool] = False


TrainingArgumentType = Union[TrainingArguments]


@dataclass
class TrainerConfig:
    trainer_type: Literal["hf_trainer"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    trainer_args: TrainingArgumentType
    trainer_args_type: Literal["sft"] = "sft"

    def to_dict(self):
        trainer_args_dict = self.trainer_args.to_dict()
        final_dict = asdict(self)
        final_dict["trainer_args"] = trainer_args_dict
        return final_dict
