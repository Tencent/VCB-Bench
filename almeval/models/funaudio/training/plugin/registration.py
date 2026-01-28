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
Register FunAudioChat plugins with LLaMA-Factory.
This includes MM plugins, attention configuration, and chat templates.
"""

import logging

logger = logging.getLogger(__name__)


def register_mm_plugins() -> None:
    """
    Register FunAudioChat multimodal plugins with LLaMA-Factory.
    """
    try:
        from llamafactory.data.mm_plugin import register_mm_plugin
        from .mm_plugins import FunAudioChatPlugin
        
        # Register FunAudioChat plugin
        try:
            register_mm_plugin("funaudiochat", FunAudioChatPlugin)
            logger.info("Registered MM plugin: funaudiochat")
        except ValueError as e:
            # Plugin already registered, skip
            logger.debug(f"MM plugin funaudiochat already registered: {e}")
        
        logger.info("Successfully registered FunAudioChat MM plugins")
        
    except ImportError as e:
        logger.error(f"Failed to import mm_plugin modules: {e}")
        logger.error("Make sure LLaMA-Factory is installed correctly")
        raise
    except Exception as e:
        logger.error(f"Failed to register MM plugins: {e}")
        raise


def register_attention_plugin() -> None:
    """
    Register FunAudioChat attention configuration plugin with LLaMA-Factory.
    This plugin handles FlashAttention-2 configuration for FunAudioChat models.
    """
    try:
        # Import the attention module
        from . import attention
        
        # Try to inject into llamafactory's model_utils
        try:
            from llamafactory.model.model_utils import attention as llama_attention
            from llamafactory.model import patcher
            
            # Get the original configure function
            original_configure = llama_attention.configure_attn_implementation
            original_print = llama_attention.print_attn_implementation
            
            # Create wrapper functions that handle both original and FunAudioChat models
            def configure_attn_implementation_wrapper(config, model_args):
                # First try FunAudioChat configuration
                if getattr(config, "model_type", None) == "funaudiochat":
                    attention.configure_attn_implementation(config, model_args)
                else:
                    # Fall back to original implementation
                    original_configure(config, model_args)
            
            def print_attn_implementation_wrapper(config):
                # First try FunAudioChat print
                if getattr(config, "model_type", None) == "funaudiochat":
                    attention.print_attn_implementation(config)
                else:
                    # Fall back to original implementation
                    original_print(config)
            
            # Replace the functions in both llamafactory.model.model_utils.attention 
            # and llamafactory.model.patcher (which imports them directly)
            llama_attention.configure_attn_implementation = configure_attn_implementation_wrapper
            llama_attention.print_attn_implementation = print_attn_implementation_wrapper
            patcher.configure_attn_implementation = configure_attn_implementation_wrapper
            patcher.print_attn_implementation = print_attn_implementation_wrapper
            
            logger.info("Registered attention plugin for FunAudioChat")
            
        except ImportError as e:
            logger.warning(f"Could not inject attention plugin into llamafactory: {e}")
            logger.warning("FunAudioChat attention configuration may not work properly")
        
    except Exception as e:
        logger.error(f"Failed to register attention plugin: {e}")
        raise


def register_templates() -> None:
    """
    Register FunAudioChat chat templates with LLaMA-Factory.
    """
    try:
        from llamafactory.data.template import register_template
        from llamafactory.data.formatter import StringFormatter
        from llamafactory.data.mm_plugin import get_mm_plugin
        
        # Check if templates already exist
        from llamafactory.data.template import TEMPLATES
        
        # Register funaudiochat template
        if "funaudiochat" not in TEMPLATES:
            try:
                register_template(
                    name="funaudiochat",
                    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
                    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
                    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
                    default_system="You are a helpful assistant.",
                    stop_words=["<|im_end|>"],
                    replace_eos=True,
                    mm_plugin=get_mm_plugin(name="funaudiochat", audio_token="<|AUDIO|>"),
                )
                logger.info("Registered template: funaudiochat")
            except ValueError as e:
                logger.debug(f"Template funaudiochat already registered: {e}")
        
        logger.info("Successfully registered all FunAudioChat templates")
        
    except ImportError as e:
        logger.error(f"Failed to import template modules: {e}")
        logger.error("Make sure LLaMA-Factory is installed correctly")
        raise
    except Exception as e:
        logger.error(f"Failed to register templates: {e}")
        raise


def register_all() -> None:
    """
    Register all FunAudioChat plugins with LLaMA-Factory.
    This is the main entry point for registration.
    """
    # Register MM plugins first
    register_mm_plugins()
    
    # Register attention plugin
    register_attention_plugin()
    
    # Then register templates
    register_templates()


__all__ = [
    "register_mm_plugins",
    "register_attention_plugin", 
    "register_templates",
    "register_all",
]

