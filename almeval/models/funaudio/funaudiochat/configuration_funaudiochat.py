# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FunAudioChat model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class FunAudioChatAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAudioChatEncoder`]. It is used to instantiate a
    Qwen2-Audio audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
    architecture.

    e.g. [FunAudioLLM/Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `FunAudioChatProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        n_window (`int`, *optional*, defaults to 100):
            Window size for chunking audio features.
        output_dim (`int`, *optional*, defaults to 3584):
            Output dimension of the encoder.
        bos_token_id (`int`, *optional*):
            Beginning of sequence token ID for the audio encoder.
        codebook_size (`int`, *optional*):
            Size of the audio codebook.
        continuous_features_mode (`str`, *optional*, defaults to "replace"):
            How to combine continuous audio features, either "add" or "replace".
        crq_transformer_config (`dict`, *optional*):
            Configuration for the CRQ transformer.
        eos_token_id (`int`, *optional*):
            End of sequence token ID for the audio encoder.
        group_size (`int`, *optional*, defaults to 5):
            Group size for audio token grouping.
        enable_audio_invert_tower (`bool`, *optional*, defaults to True):
            Whether to enable the audio invert tower.
        pad_token_id (`int`, *optional*):
            Padding token ID for the audio encoder.

    Example:

    ```python
    >>> from transformers import FunAudioChatEncoderConfig, FunAudioChatEncoder

    >>> # Initializing a FunAudioChatEncoderConfig
    >>> configuration = FunAudioChatEncoderConfig()

    >>> # Initializing a FunAudioChatEncoder (with random weights)
    >>> model = FunAudioChatEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "funaudiochat_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        bos_token_id=None,
        codebook_size=None,
        continuous_features_mode="replace",
        crq_transformer_config=None,
        eos_token_id=None,
        group_size=5,
        enable_audio_invert_tower=True,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        
        # Additional audio encoder parameters
        self.bos_token_id = bos_token_id
        self.codebook_size = codebook_size
        self.continuous_features_mode = continuous_features_mode
        self.crq_transformer_config = crq_transformer_config
        self.eos_token_id = eos_token_id
        self.group_size = group_size
        self.enable_audio_invert_tower = enable_audio_invert_tower
        self.pad_token_id = pad_token_id

class FunAudioChatConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunAudioChatForConditionalGeneration`]. It is used to instantiate an
    Qwen2-Audio model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Qwen2-Audio.

    e.g. [FunAudioLLM/Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.
        ignore_index (`int`, *optional*, defaults to -100):
            The index to ignore in loss calculation.
        hidden_size (`int`, *optional*):
            Hidden size of the model. If not specified, will use text_config.hidden_size.

    Example:

    ```python
    >>> from transformers import FunAudioChatForConditionalGeneration, FunAudioChatConfig, FunAudioChatEncoderConfig, Qwen2Config

    >>> # Initializing a FunAudioChatEncoder config
    >>> audio_config = FunAudioChatEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing a FunAudioChat configuration
    >>> configuration = FunAudioChatConfig(audio_config, text_config)

    >>> # Initializing a model from the qwen2-audio style configuration
    >>> model = FunAudioChatForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "funaudiochat"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151646,
        ignore_index=-100,
        hidden_size=None,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index
        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "funaudiochat_audio_encoder"
            )
            audio_config = FunAudioChatAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = FunAudioChatAudioEncoderConfig(
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config
        
        # Set hidden_size from text_config if not explicitly provided
        if hidden_size is None:
            self.hidden_size = self.text_config.hidden_size
        else:
            self.hidden_size = hidden_size

        super().__init__(**kwargs)


__all__ = ["FunAudioChatConfig", "FunAudioChatAudioEncoderConfig"]
