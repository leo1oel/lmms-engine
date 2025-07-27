from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    PretrainedConfig,
)
from transformers.modeling_utils import PreTrainedModel

DATASET_MAPPING = {}
DATAPROCESSOR_MAPPING = {}


# A decorator class to register processors
def register_processor(processor_type: str):
    def decorator(cls):
        if processor_type in DATAPROCESSOR_MAPPING:
            raise ValueError(f"Processor type {processor_type} is already registered.")
        DATAPROCESSOR_MAPPING[processor_type] = cls
        return cls

    return decorator


# A decorator class to register dataset
def register_dataset(dataset_type: str):
    def decorator(cls):
        if dataset_type in DATASET_MAPPING:
            raise ValueError(f"Dataset type {dataset_type} is already registered.")
        DATASET_MAPPING[dataset_type] = cls
        return cls

    return decorator


def register_model(
    model_type: str, model_config: PretrainedConfig, model_class: PreTrainedModel
):
    AutoConfig.register(model_type, model_config)
    AutoModelForCausalLM.register(model_config, model_class)


def create_model(model_name, **kwargs):
    config = AutoConfig.from_pretrained(model_name, **kwargs)

    if type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = AutoModelForCausalLM
    elif type(config) in AutoModelForVision2Seq._model_mapping.keys():
        model_class = AutoModelForVision2Seq
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model_class
