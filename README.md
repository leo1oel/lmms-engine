
# LMMs Engine

Training framework for LMMs-Lab.


## Installation
Installation is simple
```bash
python3 -m pip install -e .
```

### Use rmpad
Rmpad is a techniques to accelerate the training process by removing the pad. With it enabled, it will boost the training performance quickly. Currently the implementation is being fused with liger-kernel and being patched to the model's forward during training. Thus, we might need to validate the operations. Current Rmpad ops are all written in flash-attn with the `var_len` function so we need to install flash-attn and liger-kernel to use it. If you currently use the fully unpadding techniques start from the input ids, the MFU can reach to about 35-40 under ideal settings. Normally, in most of the cases, a range between 25-35 would be normal

#### Current Supported Ops
- Qwen2 or 2.5 LM series 
- Qwen2.5 VL
- QwenAudioEncoder
- Kino (Unpad start from input ids)

To use rmpad, you should install flash-attn also. You can do it by
```bash
python3 -m pip install flash-attn --no-build-isolation
```

If you encounter any issue for example symbol not found. This is possibly because of the flash-attn has been compiled on the wrong torch version. You can run

```bash
python3 -m pip install --no-build-isolation --no-cache-dir flash-attn
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

## Prepare Config
The overall design of our framework is that we build each component as a pipeline, you will need to pass in a config to use for init the pipeline.

An example config
```json
[
    {
        "type" : "trainer",
        "config" : {
            "trainer_type": "hf_trainer",
            "dataset_config": {
                "dataset_type" : "vision",
                "dataset_format" : "json",
                "dataset_path" : "./data/lmms_engine.json",
                "processor_config": {
                    "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
                    "processor_modality": "vision",
                    "processor_type": "qwen2_vl"
                }
            },
            "model_config": {
                "model_name_or_path" : "Qwen/Qwen2-VL-2B-Instruct",
                "model_class" : "Qwen2VLForConditionalGeneration",
                "attn_implementation" : "flash_attention_2"
            },
            "per_device_train_batch_size": 1,
            "learning_rate": 5e-05,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "save_steps": 1000,
            "report_to": "wandb",
            "output_dir": "./output",
            "warmup_ratio": 0,
            "run_name": "test_run",
            "logging_steps" : 1,
            "group_by_length" : true,
            "dataloader_num_workers" : 8,
            "bf16" : true
        }
    }
]
```

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


Launching can also be done by using `accelerate`. But somehow I find in some cases it might create separate processes if you are using multi-machine settings. This is possibly because of the settings of the machine.

```bash
# FSDP
CUDA_LAUNCH_BLOCKING=1 ACCELERATE_CPU_AFFINITY=1 accelerate launch \
    --use_fsdp \
    --mixed_precision bf16 \
    --fsdp_sharding_strategy HYBRID_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_forward_prefetch false \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_offload_params false \
    --fsdp_state_dict_type SHARDED_STATE_DICT \
    --fsdp_sync_module_states true \
    --fsdp_transformer_layer_cls_to_wrap "SiglipVisionModel,Qwen2DecoderLayer" \
    --fsdp_use_orig_params true \
    --num_processes="8" \
    --num_machines="1" \
    --main_process_ip=<port_ip> \
    --main_process_port=<port> \
    --machine_rank="0" \
    -m lmms_engine.launch.cli --config scripts/config_custom.json
```

To launch it using deepspeed, you can

```bash
CUDA_LAUNCH_BLOCKING=1 ACCELERATE_CPU_AFFINITY=1 accelerate launch \
    --use_deepspeed \
    --mixed_precision bf16 \
    --deepspeed_config_file zero3.json \
    --num_processes="8" \
    --num_machines="1" \
    --main_process_ip=<port_ip> \
    --main_process_port=<port> \
    --machine_rank="0" \
    -m lmms_engine.launch.cli --config ${CONFIG}
```

## More Content
- [Preparing Data and how the data is load](docs/data_prep.md)
- [Overall Design Principle](docs/design_principle.md)
- [Training](docs/train.md)
- [API](docs/api.md)
