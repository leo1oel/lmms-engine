import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
)

from ..aero import AeroForConditionalGeneration
from ..aero.modeling_aero import AeroCausalLMOutputWithPast
from .configuration_aero_omni import AeroOmniConfig


@dataclass
class AeroOmniCausalLMOutputWithPast(AeroCausalLMOutputWithPast):
    audio_logits: Optional[torch.FloatTensor] = None


class AeroOmniForConditionalGeneration(AeroForConditionalGeneration):
    config_class = AeroOmniConfig

    def __init__(self, config: AeroOmniConfig):
        super().__init__(config)
        # Total Vocab Size
        self.vocab_size = config.text_config.vocab_size
        # Additional Audio Vocab
        self.audio_vocab_size = config.code_book_size * config.num_codebooks
        self.audio_start_from = config.audio_token_start_from

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
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if (audio_input_ids is None) ^ (audio_inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of audio_input_ids or audio_inputs_embeds"
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
            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
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
        )

        logits = outputs[0]
        # Audio Logits should start from the vocab size
        audio_logits = logits[..., :, self.audio_start_from :].contiguous()
        logits = logits[..., :, : self.audio_start_from].contiguous()
        loss = outputs.get("loss", None)
        if labels is not None and loss is None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )
        # If codec labels is not None
        # Then we need to calculate the loss for the audio tokens
        if codec_labels is not None:
            shift_audio_logits = audio_logits[..., :-1, :].contiguous()
            shift_audio_labels = codec_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            audio_loss = loss_fct(
                shift_audio_logits.view(-1, shift_audio_logits.size(-1)),
                shift_audio_labels.view(-1).to(shift_audio_logits.device),
            )
            if loss is not None:
                loss = loss + audio_loss
            else:
                loss = audio_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AeroOmniCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            flops=flops,
            audio_hidden_states=audio_features if audio_values is not None else None,
            audio_logits=audio_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        audio_input_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        audio_values=None,
        audio_attention_mask=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        model_inputs["audio_input_ids"] = audio_input_ids

        if cache_position[0] == 0:
            model_inputs["audio_values"] = audio_values
            model_inputs["audio_attention_mask"] = audio_attention_mask

        return model_inputs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus: bool,
        streamer,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = (
                model_kwargs["past_key_values"].is_compileable
                and self._supports_static_cache
            )
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda"
                or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(
                input_ids, generation_config, **model_kwargs
            )
            is_prefill = False
        else:
            is_prefill = True

        audio_input_ids = model_kwargs.get("audio_input_ids")

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            next_audio_token_logits = outputs.audio_logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )
            next_audio_token_scores = logits_processor(
                input_ids, next_audio_token_logits
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                audio_probs = nn.functional.softmax(next_audio_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_audio_tokens = torch.multinomial(
                    audio_probs, num_samples=1
                ).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
                next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)
            next_audio_tokens = next_audio_tokens + self.audio_start_from

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            audio_input_ids = torch.cat(
                [audio_input_ids, next_audio_tokens[:, None]], dim=-1
            )
            # We only need to pass the last tokens
            model_kwargs["audio_input_ids"] = next_audio_tokens[:, None]
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, audio_input_ids
