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

"""
Attention configuration plugin for FunAudioChat model.
This plugin configures FlashAttention-2 for FunAudioChat's audio and text components.
"""

from typing import TYPE_CHECKING
import logging

try:
    from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
except ImportError:
    is_flash_attn_2_available = lambda: False
    is_torch_sdpa_available = lambda: False

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args) -> None:
    """
    Configure attention implementation for FunAudioChat model.
    
    Args:
        config: The model configuration
        model_args: Model arguments containing flash_attn setting
    """
    # Get the flash attention setting from model_args
    flash_attn = getattr(model_args, "flash_attn", "auto")
    
    # Convert string to appropriate value if needed
    if isinstance(flash_attn, str):
        flash_attn = flash_attn.lower()
    
    # Handle auto mode - default to FA2 if available
    if flash_attn == "auto":
        if is_flash_attn_2_available():
            requested_attn_implementation = "flash_attention_2"
            logger.info("Auto-selected FlashAttention-2 for FunAudioChat")
        else:
            logger.warning("FlashAttention-2 not available, using eager attention")
            return
    
    # Handle disabled mode
    elif flash_attn == "disabled" or flash_attn == "eager":
        requested_attn_implementation = "eager"
    
    # Handle SDPA mode
    elif flash_attn == "sdpa":
        if not is_torch_sdpa_available():
            logger.warning("torch>=2.1.1 is required for SDPA attention.")
            return
        requested_attn_implementation = "sdpa"
    
    # Handle FA2 mode (primary support)
    elif flash_attn == "fa2" or flash_attn == "flash_attention_2":
        if not is_flash_attn_2_available():
            logger.warning("FlashAttention-2 is not installed.")
            return
        requested_attn_implementation = "flash_attention_2"
    
    else:
        logger.warning(f"Unknown attention type: {flash_attn}, using default")
        return
    
    # Configure FunAudioChat model attention
    if getattr(config, "model_type", None) == "funaudiochat":
        # Audio encoder - only supports FA2
        if hasattr(config, "audio_config"):
            setattr(config.audio_config, "_attn_implementation", requested_attn_implementation)
            
            # Configure CRQ transformer attention if it exists
            if hasattr(config.audio_config, "crq_transformer_config") and config.audio_config.crq_transformer_config is not None:
                # setattr(config.audo_config, "crq_transformer_attn_implementation", requested_attn_implementation)
                config.audio_config.crq_transformer_config["_attn_implementation"] = requested_attn_implementation
        
        # Text config (LLM backbone)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
        
        # Main config
        setattr(config, "_attn_implementation", requested_attn_implementation)
        
        logger.info(f"Configured FunAudioChat attention: {requested_attn_implementation}")
        if hasattr(config, "audio_config"):
            logger.info(f"  - Audio encoder: {getattr(config.audio_config, '_attn_implementation', 'default')}")
        if hasattr(config, "text_config"):
            logger.info(f"  - Text config: {getattr(config.text_config, '_attn_implementation', 'default')}")


def print_attn_implementation(config: "PretrainedConfig") -> None:
    """
    Print the attention implementation being used.
    
    Args:
        config: The model configuration
    """
    attn_implementation = getattr(config, "_attn_implementation", None)
    
    if attn_implementation == "flash_attention_2":
        logger.info("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info("Using torch SDPA for faster training and inference.")
    else:
        logger.info("Using vanilla attention implementation.")
    
    # Print sub-config attention if it's a FunAudioChat model
    if getattr(config, "model_type", None) == "funaudiochat":
        if hasattr(config, "audio_config"):
            audio_attn = getattr(config.audio_config, "_attn_implementation", None)
            if audio_attn:
                logger.info(f"  - Audio encoder attention: {audio_attn}")
        if hasattr(config, "text_config"):
            text_attn = getattr(config.text_config, "_attn_implementation", None)
            if text_attn:
                logger.info(f"  - Text config attention: {text_attn}")


__all__ = ["configure_attn_implementation", "print_attn_implementation"]

