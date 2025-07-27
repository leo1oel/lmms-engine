import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.nn import CrossEntropyLoss
from transformers import Qwen2AudioEncoder
from transformers.cache_utils import StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)
from transformers.utils import logging

from lmms_engine.utils import Logging

from .configuration_qwen2_5_vl import KinoQwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from .processing_qwen2_5_vl import InputMode

logger = logging.get_logger(__name__)


class AudioMultiModalProjector(nn.Module):
    def __init__(self, config: KinoQwen2_5_VLConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.audio_config.d_model, config.hidden_size, bias=True
        )

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    flops: Optional[float] = None


class KinoQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = KinoQwen2_5_VLConfig
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.audio_tower = Qwen2AudioEncoder(config.audio_config)
        self.audio_modal_projector = AudioMultiModalProjector(config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()
        vision_lora = getattr(config, "vision_lora", None)
        self.use_vision_lora = vision_lora is not None
        peft_model = None
        if vision_lora is not None:
            vision_lora_config = LoraConfig(
                r=vision_lora["r"],
                target_modules=vision_lora["target_modules"],
                lora_alpha=vision_lora["lora_alpha"],
                lora_dropout=vision_lora["lora_dropout"],
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(
                self,
                peft_config=vision_lora_config,
                adapter_name="vision",
            )
        audio_lora = getattr(config, "audio_lora", None)
        self.use_audio_lora = audio_lora is not None
        if audio_lora is not None:
            audio_lora_config = LoraConfig(
                r=audio_lora["r"],
                target_modules=audio_lora["target_modules"],
                lora_alpha=audio_lora["lora_alpha"],
                lora_dropout=audio_lora["lora_dropout"],
                task_type="CAUSAL_LM",
            )
            if peft_model is None:
                peft_model = get_peft_model(
                    self,
                    peft_config=audio_lora_config,
                    adapter_name="audio",
                )
            else:
                peft_model.base_model.active_adapter.append("audio")
                peft_model.add_adapter("audio", audio_lora_config)
        text_lora = getattr(config, "text_lora", None)
        self.use_text_lora = text_lora is not None
        if text_lora is not None:
            text_lora_config = LoraConfig(
                r=text_lora["r"],
                target_modules=text_lora["target_modules"],
                lora_alpha=text_lora["lora_alpha"],
                lora_dropout=text_lora["lora_dropout"],
                task_type="CAUSAL_LM",
            )
            if peft_model is None:
                peft_model = get_peft_model(
                    self,
                    peft_config=text_lora_config,
                    adapter_name="text",
                )
            else:
                peft_model.base_model.active_adapter.append("text")
                peft_model.add_adapter("text", text_lora_config)
        self.use_all_adapter = (
            self.use_vision_lora and self.use_audio_lora and self.use_text_lora
        )

    def set_lora_adapter(self, adapter_name) -> None:
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        for module in self.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    Logging.warning(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
                    module.unmerge()
                module._active_adapter = adapter_name
                module._disable_adapters = False

    def unset_lora_adapter(self) -> None:
        # Ref: peft/tuners/tuners_utils.py - enable_adapters()
        # Ref: peft/tuners/lora/layer.py

        for module in self.modules():
            if isinstance(module, LoraLayer):
                module._active_adapter = []
                module._disable_adapters = True

    def prepare_dummy_pixel_inputs(self):
        channel = 3
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        hidden_dim = channel * patch_size * patch_size * temporal_patch_size
        pixel_values = torch.zeros((4, hidden_dim), requires_grad=True)
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        return pixel_values, image_grid_thw

    def prepare_audio_values(self, audio_values, audio_attention_mask):
        (
            audio_feat_lengths,
            audio_output_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(
            audio_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = audio_values.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=audio_feat_lengths.dtype,
                device=audio_feat_lengths.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(
            audio_values, attention_mask=audio_attention_mask
        )
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_modal_projector(selected_audio_feature)
        return audio_features, audio_output_lengths

    def prepare_dummy_audio_inputs(self):
        num_mel_bins = self.config.audio_config.num_mel_bins
        max_source_positions = self.config.audio_config.max_source_positions
        audio_values = torch.zeros(
            (1, num_mel_bins, max_source_positions * 2), requires_grad=True
        )
        audio_attention_mask = torch.ones(
            (1, audio_values.shape[-1]), dtype=torch.float16
        )
        return audio_values, audio_attention_mask

    def add_fake_gradient_visual(self, inputs_embeds):
        pixel_values, image_grid_thw = self.prepare_dummy_pixel_inputs()
        pixel_values = pixel_values.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        image_grid_thw = image_grid_thw.to(device=inputs_embeds.device)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += image_embeds * 0
        return inputs_embeds

    def add_fake_gradient_audio(self, inputs_embeds):
        audio_values, audio_attention_mask = self.prepare_dummy_audio_inputs()
        audio_values = audio_values.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        audio_features, audio_output_lengths = self.prepare_audio_values(
            audio_values, audio_attention_mask
        )
        inputs_embeds += audio_features.sum(dim=1) * 0
        return inputs_embeds

    def log_lora(self):
        for module in self.modules():
            if isinstance(module, LoraLayer):
                Logging.null_logging(module.lora_A.vision.weight.grad)
                Logging.null_logging(module.lora_A.audio.weight.grad)
                Logging.null_logging(module.lora_A.text.weight.grad)

    def add_fake_gradient_lora(self, hidden_states, input_mode):
        adapter_names = []
        if input_mode == InputMode.AUDIO_VISION:
            adapter_names = ["text"]
        elif input_mode == InputMode.VISION:
            adapter_names = ["text", "audio"]
        elif input_mode == InputMode.AUDIO:
            adapter_names = ["text", "vision"]
        elif input_mode == InputMode.LANGUAGE:
            adapter_names = ["vision", "audio"]
        self.set_lora_adapter(adapter_name=adapter_names)
        inputs_embeds = torch.zeros(
            (1, 1, self.config.hidden_size),
            dtype=self.model.dtype,
            device=self.model.device,
        )
        embeds_all_adapters = self.model(inputs_embeds=inputs_embeds, input_ids=None)
        hidden_states += embeds_all_adapters[0] * 0
        return hidden_states

    def get_input_mode(self, input_mode: Optional[torch.Tensor]):
        input_mode = input_mode.detach().cpu().tolist()
        if 3 in input_mode:
            input_mode = InputMode.AUDIO_VISION
        elif 2 in input_mode:
            if 1 in input_mode:
                input_mode = InputMode.AUDIO_VISION
            else:
                input_mode = InputMode.VISION
        elif 1 in input_mode:
            input_mode = InputMode.AUDIO
        elif 0 in input_mode:
            input_mode = InputMode.LANGUAGE
        return input_mode

    def set_adapter_on_input_mode(self, input_mode: InputMode):
        if input_mode == InputMode.AUDIO_VISION:
            self.set_lora_adapter(["vision", "audio"])
        elif input_mode == InputMode.VISION:
            self.set_lora_adapter("vision")
        elif input_mode == InputMode.AUDIO:
            self.set_lora_adapter("audio")
        elif input_mode == InputMode.LANGUAGE:
            self.set_lora_adapter("text")
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Only when we have all adapters, we need to set the adapter based on input_mode
        if input_mode is not None and self.use_all_adapter:
            input_mode = self.get_input_mode(input_mode)
            self.set_adapter_on_input_mode(input_mode)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if self.training and (pixel_values is None and pixel_values_videos is None):
                inputs_embeds = self.add_fake_gradient_visual(inputs_embeds)

            # Embed audio features
            if audio_values is not None:
                audio_features, audio_output_lengths = self.prepare_audio_values(
                    audio_values, audio_attention_mask
                )
                n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                n_audio_features = audio_output_lengths.sum()
                if n_audio_tokens != n_audio_features:
                    raise ValueError(
                        f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                    )
                audio_mask = (
                    (input_ids == self.config.audio_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                # Audio feature is in (bs, max_seq_len, hidden_size)
                # If directly masked scatter, the embed will be place one by one (order is incorret)
                # We remove the padded values first
                unpadded_audio_features = [
                    audio_feat[:audio_output_length]
                    for audio_feat, audio_output_length in zip(
                        audio_features, audio_output_lengths
                    )
                ]
                # Concat the audio features
                # Should exactly have audio_mask.sum() values
                unpadded_audio_features = torch.concatenate(
                    unpadded_audio_features, dim=0
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask, unpadded_audio_features
                )
            elif self.training:
                inputs_embeds = self.add_fake_gradient_audio(inputs_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                cache_position is not None and cache_position[0] == 0
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        flops = self.calc_gpt_flops(attention_mask)
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            flops=flops,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_values=None,
        audio_attention_mask=None,
        second_per_grid_ts=None,
        input_mode=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            audio_values = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {
                "input_ids": input_ids,
                "inputs_embeds": None,
                "input_mode": input_mode,
            }

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = (
                self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.lm_head.weight.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "audio_values": audio_values,
                "audio_attention_mask": audio_attention_mask,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        return model_inputs

    def flops_per_token(self):
        num_hidden_layers = self.config.num_hidden_layers
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        intermediate_size = self.config.intermediate_size
        kv_heads = self.config.num_key_value_heads
        attn_heads = self.config.num_attention_heads
        total_params = 0.0  # initilize total_params as float, to avoid overflow of 'flops' when using 512 gpus
        # head, mebedding not considered
        total_params += vocab_size * hidden_size
        # transformers
        params_per_block = 2 * hidden_size * hidden_size
        params_per_block += 4 * hidden_size * hidden_size * kv_heads // attn_heads
        params_per_block += 3 * hidden_size * intermediate_size
        total_params += params_per_block * num_hidden_layers

        flops = 6 * total_params
        return flops

    def calc_gpt_flops(self, attention_mask):
        tokens_count = torch.sum(attention_mask != 0).item()
        flops = self.flops_per_token() * tokens_count
        token_count_list = torch.sum(attention_mask != 0, dim=1).tolist()
        for seq_len in token_count_list:
            flops += (
                12
                * seq_len
                * seq_len
                * self.config.num_hidden_layers
                * self.config.hidden_size
            )
        return flops


__all__ = [
    "KinoQwen2_5_VLForConditionalGeneration",
]
