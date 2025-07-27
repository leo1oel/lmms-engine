from ..aero import AeroConfig


class AeroOmniConfig(AeroConfig):
    model_type = "aero_omni"

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_index=151648,  # These are just some dummy placeholder
        audio_pad_token_index=151649,
        audio_bos_token_index=151651,
        audio_eos_token_index=151652,
        audio_token_start_from=151650,
        projector_hidden_act="gelu",
        projector_type="mlp",
        tie_word_embeddings=False,
        code_book_size=4096,
        num_codebooks=7,
        **kwargs,
    ):
        self.code_book_size = code_book_size
        self.num_codebooks = num_codebooks
        self.audio_pad_token_index = audio_pad_token_index
        self.audio_token_start_from = audio_token_start_from
        self.audio_bos_token_index = audio_bos_token_index
        self.audio_eos_token_index = audio_eos_token_index
        super().__init__(
            text_config=text_config,
            audio_config=audio_config,
            audio_token_index=audio_token_index,
            projector_hidden_act=projector_hidden_act,
            projector_type=projector_type,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
