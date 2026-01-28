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
Sitecustomize module for auto-registering FunAudioChat plugins.

This file is automatically imported by Python at startup if it's in PYTHONPATH.
It ensures FunAudioChat plugins are registered before any training code runs.
"""

import sys
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter out media_dir warnings from LLaMA-Factory converter
class MediaDirWarningFilter(logging.Filter):
    def filter(self, record):
        return "does not exist in `media_dir`" not in record.getMessage()

# Apply filter to root logger and all existing handlers
media_filter = MediaDirWarningFilter()
logging.root.addFilter(media_filter)
for handler in logging.root.handlers:
    handler.addFilter(media_filter)

# Also add to llamafactory logger specifically (will be created later)
def add_filter_to_llamafactory():
    """Add filter to llamafactory loggers after they are created."""
    try:
        llamafactory_logger = logging.getLogger('llamafactory')
        llamafactory_logger.addFilter(media_filter)
        for handler in llamafactory_logger.handlers:
            handler.addFilter(media_filter)
    except:
        pass

# Register FunAudioChat components at Python startup
try:
    # Step 1: Register model, config, and processor with transformers
    logger.info("Registering FunAudioChat with transformers AutoClasses...")
    from funaudiochat.register import register_funaudiochat
    register_funaudiochat()
    logger.info("✓ FunAudioChatProcessor registered with AutoProcessor")
    
    # Step 2: Register MM plugins with LLaMA-Factory
    logger.info("Registering FunAudioChat plugins with LLaMA-Factory...")
    from training.plugin.registration import register_all
    register_all()
    logger.info("✓ FunAudioChat plugins registered with LLaMA-Factory")
    
    # Step 3: Apply media_dir filter to llamafactory logger
    add_filter_to_llamafactory()
    
except ImportError as e:
    # Modules not installed, will fail later with a clearer error
    logger.warning(f"Failed to import registration modules: {e}")
except Exception as e:
    logger.error(f"Failed to auto-register FunAudioChat: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)

