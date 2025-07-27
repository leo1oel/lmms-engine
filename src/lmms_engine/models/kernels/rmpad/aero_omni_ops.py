from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from lmms_engine.models.aero.modeling_aero import AeroCausalLMOutputWithPast

from .utils import _unpad_input

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except:
    print("Liger Kernel is not installed, pip install liger-kernel to use this patch")


def forward(
    self,
    input_ids: torch.LongTensor = None,
    audio_input_ids: torch.LongTensor = None,
    audio_values: torch.FloatTensor = None,
    audio_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    audio_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    codec_labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int = 0,
) -> Union[Tuple, AeroCausalLMOutputWithPast]:
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

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if (audio_input_ids is None) ^ (audio_inputs_embeds is not None):
        raise ValueError(
            "You must specify exactly one of audio_input_ids or audio_inputs_embeds"
        )

    if input_ids is None or audio_input_ids is None:
        assert (
            False
        ), "input_ids is None, please provide input_ids. To use rmpad with kino, please provide input ids. This is only used in training"

    # Unpad the input ids here
    input_ids, indices, cu_seq_lens, _ = _unpad_input(
        input_ids, attention_mask=attention_mask
    )

    audio_input_ids, _, _, _ = _unpad_input(
        audio_input_ids, attention_mask=attention_mask
    )

    # Concat the two inputs
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    if audio_input_ids is not None:
        audio_inputs_embeds = self.get_input_embeddings()(audio_input_ids)
    inputs_embeds = (inputs_embeds + audio_inputs_embeds) / 2.0

    # Embed audio features
    if audio_values is not None:
        (
            audio_feat_lengths,
            audio_output_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(
            audio_attention_mask.sum(-1)
        )
        if self.audio_tower_type == "qwen2_audio_encoder":
            inputs = self.prepare_inputs_for_qwen_audio_encoder(
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                audio_feat_lengths=audio_feat_lengths,
                audio_output_lengths=audio_output_lengths,
            )
        elif self.audio_tower_type == "qwen2_5_omni_audio_encoder":
            inputs = self.prepare_inputs_for_qwen_5_omni_audio_encoder(
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                audio_feat_lengths=audio_feat_lengths,
                audio_output_lengths=audio_output_lengths,
            )

        audio_outputs = self.audio_tower(**inputs)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_modal_projector(selected_audio_feature)
        n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
        n_audio_features = audio_output_lengths.sum()
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
            )
        audio_mask = (
            (input_ids == self.config.audio_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        if self.audio_tower_type == "qwen2_audio_encoder":
            audio_features = self.prepare_scattered_audio_values(
                audio_features, audio_output_lengths
            )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
    flops = self.calc_gpt_flops(attention_mask, n_audio_tokens)
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        cu_seq_lens=cu_seq_lens,
        indices=indices,
    )
    hidden_states = outputs.hidden_states
    logits = outputs.logits

    if labels is not None:
        labels = labels.view(-1)[indices.long()]
        shift_hidden_states = []
        shift_labels = []
        for i in range(len(cu_seq_lens) - 1):
            cur_hidden_states = hidden_states[cu_seq_lens[i] : cu_seq_lens[i + 1], :]
            cur_shift_hidden_states = cur_hidden_states[:-1, :].contiguous()
            cur_labels = labels[cu_seq_lens[i] : cu_seq_lens[i + 1]]
            cur_shift_labels = cur_labels[1:].contiguous()
            shift_hidden_states.append(cur_shift_hidden_states)
            shift_labels.append(cur_shift_labels)
        shift_hidden_states = torch.cat(shift_hidden_states, dim=0)
        shift_labels = torch.cat(shift_labels, dim=0)
        # flatten tokens
        shift_hidden_states = shift_hidden_states.view(
            -1, self.config.text_config.hidden_size
        )
        shift_labels = shift_labels.view(-1)
        reduction = "mean"
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
        text_head_weight = self.language_model.lm_head.weight[
            : self.audio_start_from, :
        ]

        loss = lce(text_head_weight, shift_hidden_states, shift_labels)
    # If codec labels is not None
    # Then we need to calculate the loss for the audio tokens
    if codec_labels is not None:
        codec_labels = codec_labels.view(-1)[indices.long()]
        shift_audio_hidden_states = []
        shift_audio_labels = []
        for i in range(len(cu_seq_lens) - 1):
            cur_hidden_states = hidden_states[cu_seq_lens[i] : cu_seq_lens[i + 1], :]
            cur_shift_hidden_states = cur_hidden_states[:-1, :].contiguous()
            cur_labels = codec_labels[cu_seq_lens[i] : cu_seq_lens[i + 1]]
            cur_shift_labels = cur_labels[1:].contiguous()
            shift_audio_hidden_states.append(cur_shift_hidden_states)
            shift_audio_labels.append(cur_shift_labels)
        shift_audio_hidden_states = torch.cat(shift_audio_hidden_states, dim=0)
        shift_audio_labels = torch.cat(shift_audio_labels, dim=0)

        # flatten tokens
        shift_audio_hidden_states = shift_audio_hidden_states.view(
            -1, self.config.text_config.hidden_size
        )
        shift_audio_labels = shift_audio_labels.view(-1)
        reduction = "mean"
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
        audio_head_weight = self.language_model.lm_head.weight[
            self.audio_start_from :, :
        ]

        audio_loss = lce(
            audio_head_weight, shift_audio_hidden_states, shift_audio_labels
        )
        if loss is not None:
            loss = loss + audio_loss
        else:
            loss = audio_loss

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return AeroCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        flops=flops,
        audio_hidden_states=audio_features if audio_values is not None else None,
    )
