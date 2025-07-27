from lmms_engine.mapping_func import register_model

from .configuration_aero_omni import AeroOmniConfig
from .modeling_aero_omni import AeroOmniForConditionalGeneration
from .processing_aero_omni import AeroOmniProcessor

register_model(
    "aero_omni",
    AeroOmniConfig,
    AeroOmniForConditionalGeneration,
)

__all__ = [
    "AeroOmniConfig",
    "AeroOmniForConditionalGeneration",
    "AeroOmniProcessor",
]
