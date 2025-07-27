from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class ModelConfig:
    model_name_or_path: str
    pretrain_mm_mlp_adapter: Optional[str] = None
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa"]] = "sdpa"
    overwrite_config: Optional[Dict[str, str]] = None
