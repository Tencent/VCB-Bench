# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.generation.logits_process import LogitsProcessorList, NoBadWordsLogitsProcessor
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import auto_docstring
from transformers.models.auto import AutoConfig, AutoModel, AutoModelForCausalLM
from .configuration_funaudiochat import FunAudioChatAudioEncoderConfig, FunAudioChatConfig


@dataclass
class FunAudioChatCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    speech_loss: Optional[torch.FloatTensor] = None
    text_logits: torch.FloatTensor = None
    speech_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None

    @property
    def logits(self) -> torch.FloatTensor:
        return self.text_logits

@auto_docstring
class FunAudioChatPreTrainedModel(PreTrainedModel):
    config_class = FunAudioChatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FunAudioChatDecoderLayer",]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = False
    _supports_static_cache = False

    def _init_weights(self, module):
        # important: this ported version of Fun-Audio-Chat isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = key.transpose(-1, -2)
    attn_weights = torch.matmul(query, key_states) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class FunAudioChatAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: FunAudioChatAudioEncoderConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,  # pass cu seq lens for FA2
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class FunAudioChatAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = FunAudioChatAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


@auto_docstring(
    custom_intro="""
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`FunAudioChatAudioEncoderLayer`].
    """
)
class FunAudioChatAudioEncoder(FunAudioChatPreTrainedModel):
    config_class = FunAudioChatAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["FunAudioChatAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList([FunAudioChatAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    @auto_docstring
    def forward(
            self,
            input_features,
            feature_lens=None,
            aftercnn_lens=None,
            speech_maxlen=None,
            **kwargs,
    ):
        r"""
        input_features (`torch.float` of shape `(features_dim, N)`): N = sum(feature_lens)
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        speech_maxlen (`int`, *optional*):
            Maximum length for the output speech embeddings. Used to pad or truncate the output to a fixed size.
        """
        # 1. 识别有效和无效的样本
        original_batch_size = feature_lens.size(0)
        device = input_features.device
        dtype = input_features.dtype

        valid_mask = feature_lens > 0
        valid_indices = torch.where(valid_mask)[0]

        # 如果所有输入的长度都为0，直接返回一个全零的张量，保持正确的形状
        if valid_indices.numel() == 0:
            output_dim = self.proj.out_features
            final_shape = (original_batch_size, speech_maxlen, output_dim)
            return BaseModelOutput(
                last_hidden_state=torch.zeros(final_shape, device=device, dtype=self.proj.weight.dtype)
            )

        # 2. 仅对有效样本进行数据准备
        # 从扁平化的 input_features 中分离出有效样本的特征
        input_features_list = input_features.split(feature_lens.tolist(), dim=1)
        valid_input_features_list = [input_features_list[i] for i in valid_indices]
        valid_input_features = torch.cat(valid_input_features_list, dim=1)

        # 筛选出有效样本的长度
        valid_feature_lens = feature_lens[valid_mask]
        valid_aftercnn_lens = aftercnn_lens[valid_mask]

        # 3. 对有效样本执行原有的处理逻辑
        # --- START: Processing logic for valid samples only ---
        chunk_num = torch.ceil(valid_feature_lens / (self.n_window * 2)).long()

        # 使用更鲁棒的方式计算 chunk_lengths
        chunk_lengths_list = []
        full_chunk_len = self.n_window * 2
        for i, length in enumerate(valid_feature_lens):
            num_chunks_for_sample = chunk_num[i].item()
            if num_chunks_for_sample == 0:
                continue
            chunk_lengths_list.extend([full_chunk_len] * (num_chunks_for_sample - 1))
            last_chunk_len = length % full_chunk_len
            if last_chunk_len == 0:
                last_chunk_len = full_chunk_len
            chunk_lengths_list.append(last_chunk_len)

        chunk_lengths = torch.tensor(chunk_lengths_list, dtype=torch.long, device=device)

        chunk_list = valid_input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
                                          : padded_embed.shape[1], :
                                          ].unsqueeze(0).to(padded_embed.dtype)

        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(valid_aftercnn_lens.tolist(), dim=0)

        # Process each audio separately to avoid memory explosion from padding
        # (e.g., 238 samples padded to 15000 length = 4.5B elements > int32 max)
        # Step 1: Apply pooling in loop using F.avg_pool1d (pure function, ZeRO-3 safe)
        pooled_list = []
        pooled_lengths = []
        for each_audio_states in hidden_states_list:
            # Apply avg pooling: (seq_len, hidden_dim) -> (hidden_dim, seq_len) -> pool -> (seq_len//2, hidden_dim)
            # Use F.avg_pool1d instead of self.avg_pooler to avoid ZeRO-3 module tracking
            # Handle very short sequences (seq_len < kernel_size) to avoid output size 0
            seq_len = each_audio_states.shape[0]
            if seq_len >= 2:
                pooled = torch.nn.functional.avg_pool1d(
                    each_audio_states.transpose(0, 1), 
                    kernel_size=2, 
                    stride=2
                ).transpose(0, 1)
            else:
                # For sequences shorter than kernel_size, skip pooling
                pooled = each_audio_states
            pooled_list.append(pooled)
            pooled_lengths.append(pooled.shape[0])

        # Step 2: Concatenate all pooled results
        pooled_concat = torch.cat(pooled_list, dim=0)  # (total_seq_len, hidden_dim)

        # Step 3: Apply ln_post and proj ONCE to avoid ZeRO-3 hang from repeated module calls
        processed_concat = self.proj(self.ln_post(pooled_concat))  # (total_seq_len, output_dim)

        # Step 4: Split back to individual audios
        processed_audio_list = list(processed_concat.split(pooled_lengths, dim=0))

        # --- END: Processing logic for valid samples only ---

        # 4. 重建完整的批次输出
        # 创建一个全零的输出张量，其 batch_size 是原始的 batch_size
        output_dim = processed_audio_list[0].shape[-1] if processed_audio_list else self.proj.out_features
        output_hidden_states = torch.zeros(
            (original_batch_size, speech_maxlen, output_dim),
            dtype=processed_audio_list[0].dtype if processed_audio_list else self.proj.weight.dtype,
            device=device,
        )

        # 将有效样本的处理结果填充到输出张量的相应位置
        for i, (valid_idx, processed) in enumerate(zip(valid_indices, processed_audio_list)):
            seq_len = min(processed.shape[0], speech_maxlen)
            output_hidden_states[valid_idx, :seq_len] = processed[:seq_len]

        return BaseModelOutput(last_hidden_state=output_hidden_states)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class FunAudioChatDiscreteEncoder(FunAudioChatPreTrainedModel):
    config_class = FunAudioChatAudioEncoderConfig

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.group_size = config.group_size
        self.hidden_size = config.output_dim
        self.continuous_features_mode = getattr(config, "continuous_features_mode", "add")  # "add", "replace"
        self.embed_tokens = nn.Embedding(config.codebook_size, self.hidden_size, self.padding_idx)
        self.output_matching = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.continual_output_matching = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        audio_ids,
        continuous_audio_features=None,
        continuous_audio_output_lengths=None,
        feature_exist_mask=None,
        return_dict=None,
    ):
        inputs_embeds = self.embed_tokens(audio_ids)
        continuous_audio_hidden_states = None

        inputs_embeds = inputs_embeds.reshape(inputs_embeds.shape[0], -1, self.group_size * self.hidden_size)
        inputs_embeds_mean_value = inputs_embeds.reshape(
            inputs_embeds.shape[0], -1, self.group_size, self.hidden_size
        ).mean(dim=2)
        hidden_states = self.output_matching(inputs_embeds_mean_value)

        if continuous_audio_features is not None:
            continuous_audio_features = continuous_audio_features.reshape(
                continuous_audio_features.shape[0], -1, self.group_size, self.hidden_size
            )
            continuous_audio_features = continuous_audio_features.mean(dim=2)
            continuous_audio_hidden_states = self.continual_output_matching(continuous_audio_features)
            if self.continuous_features_mode == "add":
                hidden_states[feature_exist_mask] += continuous_audio_hidden_states
            else:
                hidden_states[feature_exist_mask] = continuous_audio_hidden_states

        encoder_states = (inputs_embeds, hidden_states, continuous_audio_hidden_states)
        all_attentions = None
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = input_lengths
        output_lengths = (input_lengths + self.group_size - 1) // self.group_size
        return input_lengths, output_lengths


class FunAudioChatDecoder(FunAudioChatPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`FunAudioChatDecoderLayer`].

    Args:
        config: FunAudioChatAudioEncoderConfig
    """

    config_class = FunAudioChatAudioEncoderConfig
    main_input_name = "audio_ids"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FunAudioChatAudioEncoderConfig):
        super().__init__(config)
        self.group_size = config.group_size
        self.hidden_size = config.output_dim
        self.pre_matching = nn.Linear(self.hidden_size, self.hidden_size * self.group_size, bias=True)

        crq_transformer_config = AutoConfig.for_model(**config.crq_transformer_config)
        self.crq_transformer = AutoModel.from_config(crq_transformer_config)
        del self.crq_transformer.embed_tokens
        self.input_matching = nn.Linear(self.hidden_size, crq_transformer_config.hidden_size, bias=False)
        self.output_matching = nn.Linear(crq_transformer_config.hidden_size, self.hidden_size, bias=False)

        self.lm_head = nn.Linear(config.output_dim, config.codebook_size, bias=False)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_embeddings(self, audio_tokens):
        return self.lm_head.weight.data[audio_tokens]

    def sampling_step(self, logits):
        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=logits.device)

        # pre-process distribution
        next_token_scores = self.crq_logits_processor(torch.cat([self.crq_speech_ids, *self.crq_generate_tokens], dim=-1), next_token_logits)

        # token selection
        if self.crq_do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        return next_tokens, logits

    def crq_generate_forward(
        self,
        inputs_embeds=None,
        audio_embeds=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        inputs_embeds = self.pre_matching(inputs_embeds)
        bs, slen, hs = inputs_embeds.shape
        hidden_states = inputs_embeds.reshape(bs, slen * self.group_size, -1)
        self.crq_audio_embeds = (
            self.get_embeddings(self.config.bos_token_id)[None, None, :]
            .repeat(bs, 1, 1)
            .to(dtype=hidden_states.dtype, device=hidden_states.device)
            if self.crq_audio_embeds is None
            else self.crq_audio_embeds.unsqueeze(1)
        )

        # 初始化一个空列表来存储每个步骤的 logits
        all_logits = []
        self.crq_generate_tokens = []
        for i in range(self.group_size):
            if i == 0:
                input_embeds = (
                    hidden_states[:, : slen * self.group_size - (self.group_size - i - 1)] + self.crq_audio_embeds
                )
            else:
                input_embeds = (
                    hidden_states[:, slen * self.group_size - (self.group_size - i)] + self.crq_audio_embeds
                ).unsqueeze(1)
            input_embeds = self.input_matching(input_embeds)
            outputs = self.crq_transformer(
                inputs_embeds=input_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                past_key_values=self.crq_past_key_values,
                use_cache=True,
                return_dict=True,
            )
            self.crq_past_key_values = outputs.past_key_values
            lhidden_states = outputs.last_hidden_state
            lhidden_states = self.output_matching(lhidden_states)

            logits = self.lm_head(lhidden_states)
            crq_audio_tokens, logits = self.sampling_step(logits)
            self.crq_generate_tokens.append(crq_audio_tokens.unsqueeze(1))
            if i == 0:
                all_logits.append(logits)
            else:
                all_logits.append(logits[:, -1, :].unsqueeze(1))

            self.crq_audio_embeds = self.get_embeddings(crq_audio_tokens)

        self.crq_generate_tokens = torch.cat(self.crq_generate_tokens, dim=1)
        logits = torch.cat(all_logits, dim=1)
        loss = None

        # self.crq_grobal_step += 1
        all_attentions = None
        encoder_hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = (hidden_states,)
        if not return_dict:
            return tuple(v for v in [loss, logits, encoder_hidden_states, all_attentions] if v is not None)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=encoder_hidden_states, attentions=all_attentions)

    def forward(
        self,
        inputs_embeds=None,
        audio_embeds=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        inputs_embeds = self.pre_matching(inputs_embeds)
        bs, slen, _ = inputs_embeds.shape

        hidden_states = inputs_embeds.reshape(bs, slen * self.group_size, -1)

        if audio_embeds is not None:
            audio_embeds = audio_embeds.view(bs, slen * self.group_size, -1)
            audio_embeds = torch.roll(audio_embeds, shifts=-(self.group_size - 1), dims=1)
            my_inputs_embeds = hidden_states + audio_embeds
        else:
            my_inputs_embeds = hidden_states

        my_inputs_embeds = self.input_matching(my_inputs_embeds)
        
        # Expand attention_mask and position_ids for upsampling (group_size expansion)
        crq_attention_mask = None
        crq_position_ids = None
        if attention_mask is not None:
            assert attention_mask.dim() == 2, "attention_mask should be 2D"
            crq_attention_mask = attention_mask.repeat_interleave(self.group_size, dim=1)

        if position_ids is not None:
            # Expand position_ids with fine-grained offsets
            # Each position p becomes [p*group_size, p*group_size+1, ..., p*group_size+group_size-1]
            # Example: [0, 1, 2, 3] with group_size=4 -> [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
            # [bs, slen] -> [bs, slen, group_size] -> [bs, slen * group_size]
            base_positions = position_ids * self.group_size  # Scale by group_size
            crq_position_ids = base_positions.unsqueeze(-1).repeat(1, 1, self.group_size)
            # Add sub-position offsets within each group
            offsets = torch.arange(self.group_size, device=position_ids.device, dtype=position_ids.dtype)
            crq_position_ids = crq_position_ids + offsets.view(1, 1, -1)
            crq_position_ids = crq_position_ids.view(bs, -1)
            
            # Determine padding mask from position_ids (rightmost non-zero position)
            # instead of attention_mask (which may be all 1s in non-neat_packing mode)
            nonzero_mask = position_ids != 0  # [bs, slen]
            reversed_nonzero = torch.flip(nonzero_mask, dims=[-1])  # [bs, slen]
            first_nonzero_from_end = reversed_nonzero.to(torch.long).argmax(dim=-1)  # [bs]
            has_nonzero = nonzero_mask.any(dim=-1)  # [bs]
            orig_slen = position_ids.shape[1]
            ending_pos = torch.where(has_nonzero, orig_slen - first_nonzero_from_end, torch.zeros_like(first_nonzero_from_end))
            seq_indices = torch.arange(orig_slen, device=position_ids.device).unsqueeze(0)  # [1, slen]
            padding_mask = seq_indices >= ending_pos.unsqueeze(-1)  # [bs, slen]
            crq_padding_mask = padding_mask.repeat_interleave(self.group_size, dim=1)  # [bs, slen * group_size]
            crq_position_ids = crq_position_ids.masked_fill(crq_padding_mask, 0)
        
        outputs = self.crq_transformer(
            inputs_embeds=my_inputs_embeds,
            attention_mask=crq_attention_mask,
            position_ids=crq_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.output_matching(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = nn.functional.pad(labels, (0, self.group_size), value=-100)
            shift_labels = labels[..., self.group_size :].contiguous()
            loss = self.loss_function(logits, labels, vocab_size=self.config.codebook_size, shift_labels=shift_labels)

        all_attentions = None
        encoder_hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = (hidden_states,)
        if not return_dict:
            return tuple(v for v in [loss, logits, encoder_hidden_states, all_attentions] if v is not None)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=encoder_hidden_states, attentions=all_attentions)


class FunAudioChatForConditionalGeneration(FunAudioChatPreTrainedModel, GenerationMixin):
    def __init__(self, config: FunAudioChatConfig):
        super().__init__(config)
        self.continuous_audio_tower = FunAudioChatAudioEncoder(config.audio_config)
        self.audio_tower = FunAudioChatDiscreteEncoder(config.audio_config)
        if getattr(config.audio_config, "enable_audio_invert_tower", True):
            self.audio_invert_tower = FunAudioChatDecoder(config.audio_config)
        else:
            self.audio_invert_tower = None

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.text_pad_token_id = (
            self.config.text_config.pad_token_id if self.config.text_config.pad_token_id is not None else -1
        )
        self.audio_pad_token_id = (
            self.config.audio_config.pad_token_id if self.config.audio_config.pad_token_id is not None else -1
        )
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.sp_gen_kwargs = {
            'text_greedy': False,
            'only_crq_sampling': True,
            'disable_speech': False,
            'force_text_abos': False,
        }
        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        # audio tie weights
        if self.audio_invert_tower is not None:
            self._tie_or_clone_weights(self.audio_invert_tower.lm_head, self.audio_tower.embed_tokens)
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        speech_maxlen: Optional[int] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = self.continuous_audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.continuous_audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
            speech_maxlen=speech_maxlen,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features, audio_output_lengths

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        speech_ids: Optional[torch.LongTensor] = None,
        text_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        feature_exist_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FunAudioChatCausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. These tokens can be text tokens or special tokens
                like the audio token placeholder.
            input_features (`torch.FloatTensor` of shape `(audio_batch_size, num_mel_bins, feature_sequence_length)`, *optional*):
                Continuous audio features (e.g., Mel spectrograms) used by the continuous audio tower.
            speech_ids (`torch.LongTensor` of shape `(audio_batch_size, speech_sequence_length)`, *optional*):
                Indices of discrete audio tokens from a speech tokenizer/codec. These are encoded by the discrete audio
                encoder.
            text_ids (`torch.LongTensor` of shape `(audio_batch_size, speech_sequence_length)`, *optional*):
                A separate tensor containing the tokenized transcripts of the audio segments. These tokens are not part
                of the main `input_ids` sequence. Instead, their embeddings are fused with audio features to create
                text-aware representations that replace the audio placeholder tokens in the language model's input.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices in `input_ids`.
            speech_attention_mask (`torch.Tensor` of shape `(audio_batch_size, speech_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding tokens in `speech_ids`.
            feature_attention_mask (`torch.Tensor` of shape `(audio_batch_size, feature_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding in `input_features`.
            feature_exist_mask (`torch.Tensor` of shape `(audio_batch_size,)`, *optional*):
                A boolean mask indicating which samples in the batch have continuous audio features, used for mixed
                data processing.
            text_attention_mask (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                A sample-level boolean mask indicating which audio segments have a corresponding text transcript. If
                `False` for a sample, its text features are ignored.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            past_key_values (`Cache`, *optional*):
                Contains pre-computed key and value hidden-states of the attention blocks. Used to speed up decoding.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss. For text tokens, it's the next token ID. For audio
                placeholder tokens, the corresponding `speech_ids` are used to compute the audio reconstruction loss.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            A `Union[Tuple, FunAudioChatCausalLMOutputWithPast]`.

            - If `return_dict` is `False`, returns a `tuple` where the first element is the total loss.
            - If `return_dict` is `True`, returns an `FunAudioChatCausalLMOutputWithPast` object with the following fields:
                - `loss` (`torch.FloatTensor` of shape `(1,)`, *optional*): Total loss, which is the sum of `text_loss` and `speech_loss`.
                - `text_loss` (`torch.FloatTensor` of shape `(1,)`, *optional*): The Causal Language Modeling loss for text tokens.
                - `speech_loss` (`torch.FloatTensor` of shape `(1,)`, *optional*): The loss for reconstructing the discrete audio tokens.
                - `text_logits` (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`): Prediction scores of the language modeling head.
                - `speech_logits` (`torch.FloatTensor` of shape `(batch_size, sequence_length, speech_vocab_size)`, *optional*): Prediction scores for the discrete audio token reconstruction.
                - `past_key_values` (`List[torch.FloatTensor]`, *optional*): Contains pre-computed key and value hidden-states of the attention blocks.
                - `hidden_states` (`Tuple[torch.FloatTensor]`, *optional*): Tuple of `torch.FloatTensor` (one for the output of the embeddings, if applicable, and one for the output of each layer).
                - `attentions` (`Tuple[torch.FloatTensor]`, *optional*): Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
                - `attention_mask` (`torch.FloatTensor`, *optional*): The attention mask used for the forward pass.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        speech_labels = None
        audio_features = None
        text_features = None
        audio_embeds = None

        if text_ids is not None and text_ids.numel() == 0:
            text_ids = None

        target_device = self.audio_tower.device

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if speech_ids is not None and input_ids.shape[1] != 1:
                speech_ids = speech_ids.to(device=target_device, dtype=input_ids.dtype)
                speech_attention_mask = speech_attention_mask.to(target_device)
                mid_speech_labels = torch.where(speech_attention_mask == 1, speech_ids, self.config.ignore_index)
                # 将长度扩展到5的倍数
                speech_padding_target_length = (
                    (speech_ids.shape[-1] + self.audio_tower.group_size - 1) // self.audio_tower.group_size
                ) * self.audio_tower.group_size
                if speech_padding_target_length > speech_ids.shape[-1]:
                    padding_length = speech_padding_target_length - speech_ids.shape[-1]
                    speech_ids = torch.nn.functional.pad(
                        speech_ids, (0, padding_length), value=self.audio_pad_token_id
                    )
                    mid_speech_labels = torch.nn.functional.pad(
                        mid_speech_labels, (0, padding_length), value=self.config.ignore_index
                    )
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    speech_attention_mask.sum(-1)
                )

                continuous_audio_features = None
                continuous_audio_output_lengths = None
                if input_features is not None and hasattr(self, "continuous_audio_tower"):
                    continuous_audio_features, continuous_audio_output_lengths = self.get_audio_features(
                        input_features.to(target_device),
                        feature_attention_mask=feature_attention_mask.to(target_device),
                        speech_maxlen=speech_ids.shape[-1],
                    )
                    continuous_audio_features = continuous_audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

                audio_features, *_ = self.audio_tower(
                    speech_ids,
                    continuous_audio_features=continuous_audio_features,
                    continuous_audio_output_lengths=continuous_audio_output_lengths,
                    feature_exist_mask=feature_exist_mask,
                )

                mid_text_labels = None
                # Combine text of audio to audio features
                if text_ids is not None:
                    # 将长度与audio对齐
                    max_audio_output_length = int(speech_padding_target_length // self.audio_tower.group_size)
                    if text_ids.shape[1] > max_audio_output_length:
                        text_ids = text_ids[:, :max_audio_output_length]
                    elif text_ids.shape[1] < max_audio_output_length:
                        text_ids = torch.nn.functional.pad(
                            text_ids,
                            (0, max_audio_output_length - text_ids.shape[1]),
                            value=self.config.text_config.pad_token_id or self.config.text_config.eos_token_id,
                        )
                    text_ids[text_ids == self.config.text_config.eos_token_id] = self.config.text_config.sil_index
                    if self.config.text_config.pad_token_id is not None:
                        text_ids[text_ids == self.config.text_config.pad_token_id] = self.config.text_config.sil_index
                    text_ids = text_ids.to(device=target_device, dtype=input_ids.dtype)
                    mid_text_labels = torch.where(
                        speech_attention_mask[:, :: self.config.audio_config.group_size] == 1,
                        text_ids,
                        self.config.ignore_index,
                    )

                    text_features = self.get_input_embeddings()(text_ids)
                    text_features = text_features.masked_fill(~text_attention_mask[:, None, None], 0.0).to(
                        dtype=text_features.dtype
                    )
                    mid_text_labels.masked_fill_(~text_attention_mask[:, None], self.config.ignore_index)
                    mid_text_labels = mid_text_labels.to(dtype=input_ids.dtype)

                    audio_features = (text_features + audio_features) / 2

                num_audios, max_audio_tokens, embed_dim = audio_features.shape
                audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
                audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                audio_features = audio_features[audio_features_mask]

                n_audio_tokens = (input_ids == self.config.audio_token_index).sum().item()
                n_audio_features = audio_features.shape[0]

                if n_audio_tokens != n_audio_features:
                    raise ValueError(
                        f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                    )
                special_audio_mask = (input_ids == self.config.audio_token_index).to(inputs_embeds.device)
                special_audio_mask = special_audio_mask.unsqueeze(-1)
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_audio_mask.expand_as(inputs_embeds), audio_features
                )

                # 开始构建labels
                if labels is not None:
                    # Speech LLM
                    speech_labels = torch.full(
                        (*labels.shape, self.audio_tower.group_size),
                        self.config.ignore_index,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    mid_speech_labels = mid_speech_labels.view(
                        mid_speech_labels.shape[0], -1, self.audio_tower.group_size
                    )
                    mid_speech_labels = mid_speech_labels[
                        audio_features_mask.unsqueeze(-1).expand_as(mid_speech_labels)
                    ]
                    speech_labels = speech_labels.masked_scatter(
                        special_audio_mask.expand_as(speech_labels), mid_speech_labels
                    )
                    non_labels_mask = labels == self.config.ignore_index
                    speech_labels = speech_labels.masked_fill(non_labels_mask.unsqueeze(-1), self.config.ignore_index)

                    audio_eos_mask = labels == self.config.text_config.audio_eos_index
                    speech_labels = speech_labels.masked_fill(
                        audio_eos_mask.unsqueeze(-1), self.config.audio_config.eos_token_id
                    )
                    speech_labels = speech_labels.reshape(speech_labels.shape[0], -1)

                    # Text LLM
                    labels[labels == self.config.audio_token_index] = self.config.ignore_index
                    if mid_text_labels is not None:
                        mid_text_labels = mid_text_labels[audio_features_mask]
                        labels = labels.masked_scatter(special_audio_mask.squeeze(-1), mid_text_labels)
                        labels.masked_fill_(non_labels_mask, self.config.ignore_index)


        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        aux_loss = outputs.aux_loss if hasattr(outputs, 'aux_loss') else None
        speech_loss = None
        speech_logits = None
        if self.audio_invert_tower is not None:
            last_hidden_state = outputs.hidden_states[-1]

            if speech_labels is not None:
                audio_embeds = self.audio_tower(
                    speech_labels.masked_fill(speech_labels == self.config.ignore_index, self.audio_pad_token_id),
                    return_dict=True,
                ).hidden_states[0]
            else:
                audio_embeds = None

            if not self.sp_gen_kwargs['disable_speech']:
                speech_inputs_embeds = last_hidden_state
                if text_embeds is None:
                    text_embeds = self.get_input_embeddings()(input_ids)
                    if text_features is not None:
                        text_embeds = text_embeds.masked_scatter(special_audio_mask.expand_as(text_embeds), text_features[audio_features_mask])
                speech_inputs_embeds = speech_inputs_embeds + text_embeds.detach()
                speech_output = self.audio_invert_tower(
                    audio_embeds=audio_embeds,
                    inputs_embeds=speech_inputs_embeds,
                    labels=speech_labels if labels is not None else None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
                speech_loss = speech_output.loss
                speech_logits = speech_output.logits

        loss = None
        text_loss = None
        if labels is not None:
            text_loss = outputs.loss
            loss = text_loss + 0.0

            if speech_loss is not None:
                loss += speech_loss

        if not return_dict:
            output = (outputs.logits, speech_logits) + outputs[1:]
            return (loss, text_loss, speech_loss) + output if loss is not None else output

        return FunAudioChatCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            text_loss=text_loss,
            speech_loss=speech_loss,
            text_logits=outputs.logits,
            speech_logits=speech_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        speech_ids = kwargs.pop("speech_ids", None)
        single_modal = kwargs.pop("single_modal", False)
        speech_attention_mask = kwargs.pop("speech_attention_mask", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        if not self.is_prefill:
            text_features = self.get_input_embeddings()(input_ids[:, -1]).unsqueeze(1)
            model_inputs.update({"text_embeds": text_features})

        if not any(self.generate_speech) and self.audio_invert_tower is not None:
            self.audio_invert_tower.crq_grobal_step = 0

        # 9. Deal with speech generation
        self.generate_speech |= input_ids[:, -1] == self.config.text_config.audio_bos_index
        if any(self.generate_speech) and speech_ids is not None and speech_ids.shape[-1] != 0:
            audio_features = self.audio_tower(speech_ids[:, -self.config.audio_config.group_size:])[0]
            text_features = self.get_input_embeddings()(input_ids[:, -1]).unsqueeze(1)
            # XXX: double generate
            if not single_modal:
                audio_features = (text_features + audio_features) / 2
            inputs_embeds = torch.where(self.generate_speech[:, None].unsqueeze(1), audio_features, text_features)
            model_inputs.update({"input_ids": None, "inputs_embeds": inputs_embeds})

        model_inputs.update(
            {
                "speech_attention_mask": speech_attention_mask,
                "speech_ids": speech_ids,
            }
        )
        return model_inputs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: "LogitsProcessorList",
        stopping_criteria: "StoppingCriteriaList",
        generation_config: "GenerationConfig",
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union["GenerateNonBeamOutput", torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        text_greedy = self.sp_gen_kwargs.get('text_greedy', False)
        only_crq_sampling = self.sp_gen_kwargs.get('only_crq_sampling', False)
        force_text_abos = self.sp_gen_kwargs.get('force_text_abos', False)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        # CH: Generate speech init
        speech_ids = torch.empty(input_ids.size(0), 0, dtype=input_ids.dtype, device=input_ids.device)
        self.generate_speech = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        self.audio_invert_tower.forward = self.audio_invert_tower.crq_generate_forward
        self.audio_invert_tower.crq_past_key_values = None
        self.audio_invert_tower.crq_audio_embeds = None
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_speech_ids = speech_ids
        # Filter out NoBadWordsLogitsProcessor for crq (audio generation)
        crq_logits_processor = LogitsProcessorList([p for p in logits_processor if not isinstance(p, NoBadWordsLogitsProcessor)])
        self.audio_invert_tower.crq_logits_processor = crq_logits_processor
        self.audio_invert_tower.crq_grobal_step = 0

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            self.is_prefill = is_prefill
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

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
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            if not text_greedy:
                next_token_scores = logits_processor(input_ids, next_token_logits)
            else:
                next_token_scores = next_token_logits

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
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
            if do_sample and not text_greedy:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if force_text_abos:
                next_tokens = torch.zeros_like(next_tokens) + self.config.text_config.audio_bos_index
                force_text_abos = False

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # CH: Add speech decoding
            if any(self.generate_speech) and not self.sp_gen_kwargs["disable_speech"]:
                if only_crq_sampling:
                    next_speech_tokens = self.audio_invert_tower.crq_generate_tokens
                else:
                    next_speech_token_logits = outputs.speech_logits.clone()[
                        :, -self.config.audio_config.group_size :, :
                    ].float()
                    next_speech_token_logits = next_speech_token_logits.to(input_ids.device)

                    # pre-process distribution
                    next_speech_token_scores = logits_processor(
                        speech_ids.view(speech_ids.size(0) * self.config.audio_config.group_size, -1),
                        next_speech_token_logits.view(-1, next_speech_token_logits.size(-1)),
                    )
                    if do_sample:
                        speech_probs = nn.functional.softmax(next_speech_token_scores, dim=-1)
                        next_speech_tokens = torch.multinomial(speech_probs, num_samples=1)
                    else:
                        next_speech_tokens = torch.argmax(next_speech_token_scores, dim=-1)
                    next_speech_tokens = next_speech_tokens.view(-1, self.config.audio_config.group_size)

                finish_speech = (next_speech_tokens == self.config.audio_config.eos_token_id).any(dim=-1)

                combined_next_speech_tokens = torch.where(
                    finish_speech[:, None], self.config.audio_config.eos_token_id, next_speech_tokens
                )

                if streamer is not None:
                    streamer.put(combined_next_speech_tokens.cpu())

                speech_ids = torch.cat([speech_ids, combined_next_speech_tokens], dim=-1)
                self.audio_invert_tower.crq_speech_ids = speech_ids
                model_kwargs["speech_ids"] = speech_ids

                # input_ids[:, -1].masked_fill_(self.generate_speech, self.config.audio_token_index)
                input_ids[:, -1].masked_fill_(finish_speech, self.config.text_config.audio_eos_index)

                self.generate_speech = torch.logical_and(self.generate_speech, ~finish_speech)
            else:
                model_kwargs["speech_ids"] = None

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        return input_ids, speech_ids


__all__ = [
    "FunAudioChatForConditionalGeneration",
    "FunAudioChatPreTrainedModel",
    "FunAudioChatAudioEncoder",
]
