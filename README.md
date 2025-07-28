
# LMMs Engine

Training framework for LMMs-Lab.


## Installation
Installation is simple
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### Sequence Packing
Sequence packing is a techniques to accelerate the training process by removing the pad. With it enabled, it will boost the training performance quickly. Currently the implementation is being fused with liger-kernel and being patched to the model's forward during training. Thus, we might need to validate the operations. Current sequence packing ops are all written in flash-attn with the `var_len` function so we need to install `flash-attn` and `liger-kernel` to use it. If you currently use the fully unpadding techniques start from the input ids, the MFU can reach to about 35-40 under ideal settings. Normally, in most of the cases, a range between 25-35 would be normal

#### Current Supported Ops
- Qwen2 or 2.5 LM series 
- Qwen2.5 VL
- QwenAudioEncoder

To use rmpad, you should install flash-attn also. You can do it by
```bash
uv pip install flash-attn --no-build-isolation
```

If you encounter any issue for example symbol not found. This is possibly because of the flash-attn has been compiled on the wrong torch version. You can run

```bash
uv pip install --no-build-isolation --no-cache-dir flash-attn
```

To use it, you will need to set
```json
{
    ...
    "use_liger_kernel": true,
    "use_rmpad": true
}
```
in the training config. Then the forward would be patched into the model.

#### Debugging Advise

If you are trying to debug the forward function during training, you need to go into the kernels and edit the code there. Otherwise, the original forward function will be patched and would not be affected.


### Liger Kernel
[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput and reduces memory usage. Based on my testing, it does reduces memory usage when finetuning models. Benchmarking based on my testing under kino stage-1 training settings, it reduces the memory usage by around 30%. The major memory reduction is on the fused CrossEntropy kernel and allow us to use large batch size during training.

To use it is simple, you need to first install it using `pip install liger-kernel`. Then set the `use_liger_kernel` in the trainer config to `true`. The patching logic currently is as follows:

1. For our custom model, you will need to write your own `apply_liger_kernel_to_xxx` and register the model type to the `MODEL_REGISTRY` in the monkey patch. 
2. If the model is not in the registry, we will search if it is in the original liger-kernel implementation
3. If the model is not in the registry, we will see if it contains a `language_model` component and apply liger-kernel on that

## Launch

The recommended way to launch is always use torchrun as it is the most native way to launch torch and in most of the settings this should work. Most of the debug and development should be based on this as we might not always use accelerate in our later framework.

```bash
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="<port_ip>" \
    --master_port="<port>" \
    -m lmms_engine.launch.cli --config ${CONFIG}
```

## Examples

We provide two examples here to demonstrate how to use the training engine in most of the case, you will need to perform the following three steps:
1. Process the dataset into a specific format and store it in (jsonl/json/arrow)
2. Write your dataset yaml (Optional if you are only using a single data source)
3. Prepare your training config

### 1. Process the dataset
You will need to process the dataset in OpenAI chat messages format. We prepare an example for you to reference. You can get the data by using

```bash
huggingface-cli download kcz358/open-thoughts-debug --local-dir data/open_thoughts_debug --repo-type dataset
```

### 2. Prepare dataset yaml
You can specify the data by using the following yaml, data folder can be left empty for text dataset.
```yaml
datasets:
- path: data/open_thoughts_debug
  data_folder: ""
  data_type: arrow
```

### 3. Prepare training config
The last step would be to prepare the training config. We support fsdp2 and deepspeed zero

```json
[
    {
        "type" : "trainer",
        "config" : {
            "trainer_type": "hf_trainer",
            "dataset_config": {
                "dataset_type" : "vision",
                "dataset_format" : "yaml",
                "dataset_path" : "./scripts/data_yaml/debug.yaml",
                "packing": true,
                "packing_strategy": "first_fit",
                "packing_length": 20480,
                "processor_config": {
                    "processor_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "processor_type": "qwen2_5_vl"
                }
            },
            "model_config": {
                "model_name_or_path" : "Qwen/Qwen2.5-VL-7B-Instruct",
                "attn_implementation" : "flash_attention_2"
            },
            "per_device_train_batch_size": 1,
            "learning_rate": 1e-06,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": true,
            "num_train_epochs": 1,
            "save_steps": 100,
            "save_total_limit" : 1,
            "report_to": "none",
            "output_dir": "./output/debug",
            "warmup_ratio": 0.0,
            "run_name": "qwen2_5_vl_mix_of_thoughts",
            "eval_strategy": "no",
            "logging_steps" : 1,
            "group_by_length" : true,
            "dataloader_num_workers" : 8,
            "bf16" : true,
            "lr_scheduler_type" : "cosine",
            "freeze_modules" : ["visual"],
            "use_liger_kernel": true,
            "use_rmpad": true,
            "fsdp2": true,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],
                "reshard_after_forward": false
            }
        }
    }
]
```

to switch from fsdp2 to zero2, you can simply remove the fsdp tag and add
```json
{
    ...
    "deepspeed" : "path/zero2.json"
}
```


## More Content
- [Preparing Data and how the data is load](docs/data_prep.md)
- [Overall Design Principle](docs/design_principle.md)
- [Training](docs/train.md)
- [API](docs/api.md)
