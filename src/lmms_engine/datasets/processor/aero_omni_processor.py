from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from lmms_engine.mapping_func import register_processor
from lmms_engine.models.aero.processing_aero import AeroProcessorKwargs
from lmms_engine.models.aero_omni import AeroOmniProcessor

from .aero_processor import AeroDataProcessor


@register_processor("aero_omni")
class AeroOmniDataProcessor(AeroDataProcessor):
    def _build_processor(self) -> AeroOmniProcessor:
        return AeroOmniProcessor.from_pretrained(self.config.processor_name)

    def process(
        self,
        images: List[Image.Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        add_system_prompt=True,
        **kwargs,
    ):
        """
        A wrapper method to process single data
        """

        output_kwargs = self.processor._merge_kwargs(
            AeroProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        audio_inputs = {}

        if audios is not None:
            audio_inputs = self.processor.audio_processor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
                **kwargs,
            )
            audio_inputs["audio_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            audio_inputs["audio_values"] = audio_inputs.pop("input_features")
            input_lengths = (audio_inputs["audio_attention_mask"].sum(-1) - 1) // 2 + 1
            num_audio_tokens = (input_lengths - 2) // 2 + 1
        else:
            num_audio_tokens = None

        inputs = self.get_qwen_template_labels(
            hf_messages,
            num_audio_tokens,
            add_system_prompt=add_system_prompt,
        )
        if audios is not None:
            inputs["audio_values"] = audio_inputs["audio_values"]
            inputs["audio_attention_mask"] = audio_inputs["audio_attention_mask"]

        return inputs

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_audio_tokens: List[int],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
    ):
        special_tokens = self.processor.tokenizer.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [
            self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        audio_input_id, codec_label = [], []
        # The purpose of start from is to record which mm token we are at. Supposing the format is interleaved
        # Then we need to record this so that the mm token can be expanded correctly per conversation
        # If the format is not interleaved, then nothing special (Say always at the from). Start from does not matter
        image_start_from = 0
        audio_start_from = 0
        video_start_from = 0

        if add_system_prompt:
            input_id += self.processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}]
            )
            target += [-100] * len(input_id)
            # For system part, we do padding on audio stream
            audio_input_id += [self.audio_pad_token_id] * len(input_id)
            codec_label += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            # Cautions, qwen2_5 vl tokenizer wrap into a list
            encode_id = self.processor.apply_chat_template([message], tokenize=True)[0]
            if self.audio_token_id in encode_id:
                encode_id, used_audio = self._expand_encode_id_audio_tokens(
                    encode_id, num_audio_tokens, audio_start_from
                )
                audio_start_from += used_audio
            input_id += encode_id

            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
                # When it is not assistant, still don't speak
                audio_input_id += [self.audio_pad_token_id] * len(encode_id)
                codec_label += [-100] * len(encode_id)
            else:
                # Adopted from llava-ov that mask out the assistant
                encode_id[:3] = [-100] * 3
                target += encode_id
                # When it is assistant, hardcode the first 3 for
                # <|im_start|>assistant\n
                audio_input_id += [self.audio_pad_token_id] * 3
                codec_label += [-100] * 3
                # Assuming that the assistant all text
                for cont in message["content"]:
                    audio_token = cont["audio_text"]
                    # Delay = 1
                    audio_input_id += [self.audio_bos_token_id]
                    audio_input_id += audio_token
                    # All audio tokens need a shift because we start from (text, audio_pad, audio1, ...)
                    codec_label += [
                        self.audio_bos_token_id - self.audio_pad_token_id - 1
                    ]
                    # Because we splitted the lm head when calculating loss
                    # So we adjust the position of audio token
                    audio_token = [a - self.audio_pad_token_id - 1 for a in audio_token]
                    codec_label += audio_token
                    audio_input_id += [self.audio_eos_token_id]
                    codec_label += [
                        self.audio_eos_token_id - self.audio_pad_token_id - 1
                    ]

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == self.audio_token_id:
                target[idx] = -100

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        audio_input_id = torch.tensor(audio_input_id, dtype=torch.long)
        codec_label = torch.tensor(codec_label, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
            audio_input_ids=audio_input_id,
            codec_labels=codec_label,
        )

    @property
    def audio_pad_token_id(self):
        audio_pad_token = getattr(self.processor, "audio_pad_token", None)
        if audio_pad_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.audio_pad_token
            )

    @property
    def audio_bos_token_id(self):
        audio_bos_token = getattr(self.processor, "audio_bos_token", None)
        if audio_bos_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.audio_bos_token
            )

    @property
    def audio_eos_token_id(self):
        audio_eos_token = getattr(self.processor, "audio_eos_token", None)
        if audio_eos_token is None:
            return None
        else:
            return self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.audio_eos_token
            )
