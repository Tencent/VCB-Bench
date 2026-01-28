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
Custom streamer definition for FunaudioChat s2s model
"""
import torch
from transformers.generation.streamers import BaseStreamer
import re

from funaudiochat.register import register_funaudiochat
register_funaudiochat()

class FunaudioChatStreamer(BaseStreamer):
    """
    Custom Streamer for streaming text_ids and audio_ids
    
    The model will call put() twice:
        1. streamer.put(next_tokens) - Text tokens, shape (batch_size,)
        2. streamer.put(next_speech_tokens) - Audio tokens, shape (batch_size, group_size)
    """
    def __init__(self, processor, skip_prompt=True, group_size=5, **decode_kwargs):
        self.processor = processor
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.group_size = group_size
        
        self.step_results = []
        self.text_token_cache = []
        self.audio_token_cache = []
        self.pending_text_ids = None
        self.pending_text_str = None
        
        self.done = False
        self.prompt_length = 0
        
    def put(self, value):
        """
        Receiving generated tokens
        """
        is_text_token = False
        is_audio_token = False
        
        if value.dim() == 1:
            is_text_token = True
            text_ids = value.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        elif value.dim() == 2:
            if value.shape[1] == self.group_size or value.shape[1] > 1:
                is_audio_token = True
                audio_ids = value
            else:
                is_text_token = True
                text_ids = value
        
        # handle text tokens
        if is_text_token:
            # mark the prompt length for the first call
            if len(self.text_token_cache) == 0 and self.skip_prompt and text_ids.shape[1] > 1:
                self.prompt_length = text_ids.shape[1] - 1
                new_text_ids = text_ids[:, -1:]  
            else:
                new_text_ids = text_ids
            
            self.text_token_cache.append(new_text_ids.clone())
            
            new_text_str = ""
            try:
                new_text_str = self.processor.decode(new_text_ids[0], **self.decode_kwargs)
            except Exception as e:
                new_text_str = f"[Error: {e}]"
            
            self.pending_text_ids = new_text_ids.clone()
            self.pending_text_str = new_text_str
        
        # handle audio tokens
        elif is_audio_token:
            self.audio_token_cache.append(audio_ids.clone())
            
            new_audio_str = ""
            try:
                new_audio_str = self.processor.speech_tokenizer.decode(audio_ids[0])
            except Exception as e:
                new_audio_str = f"[Error: {e}]"
            
            step_result = {
                'step': len(self.step_results) + 1,
                'new_text_ids': self.pending_text_ids,
                'new_audio_ids': audio_ids.clone(),
                'new_text_str': self.pending_text_str if self.pending_text_str else "",
                'new_audio_str': new_audio_str,
            }
            self.step_results.append(step_result)
            
            self.pending_text_ids = None
            self.pending_text_str = None
        else:
            if self.pending_text_ids is not None:
                step_result = {
                    'step': len(self.step_results) + 1,
                    'new_text_ids': self.pending_text_ids,
                    'new_audio_ids': None,
                    'new_text_str': self.pending_text_str if self.pending_text_str else "",
                    'new_audio_str': "",
                }
                self.step_results.append(step_result)
                self.pending_text_ids = None
                self.pending_text_str = None
    
    def end(self):
        if self.pending_text_ids is not None:
            step_result = {
                'step': len(self.step_results) + 1,
                'new_text_ids': self.pending_text_ids,
                'new_audio_ids': None,
                'new_text_str': self.pending_text_str if self.pending_text_str else "",
                'new_audio_str': "",
            }
            self.step_results.append(step_result)
            self.pending_text_ids = None
            self.pending_text_str = None
        
        self.done = True
    
    def get_step_results(self):
        return self.step_results
    
    def get_latest_step(self):
        if len(self.step_results) > 0:
            return self.step_results[-1]
        return None
    
    def get_accumulated_results(self):
        if len(self.text_token_cache) > 0:
            full_text_ids = torch.cat(self.text_token_cache, dim=1)
            try:
                full_text_str = self.processor.decode(full_text_ids[0], **self.decode_kwargs)
            except:
                full_text_str = ""
        else:
            full_text_ids = torch.tensor([[]], dtype=torch.long)
            full_text_str = ""
            
        if len(self.audio_token_cache) > 0:
            full_audio_ids = torch.cat(self.audio_token_cache, dim=1)
            try:
                full_audio_str = self.processor.speech_tokenizer.decode(full_audio_ids[0])
            except:
                full_audio_str = ""
        else:
            full_audio_ids = torch.tensor([[]], dtype=torch.long)
            full_audio_str = ""
            
        return {
            'text_ids': full_text_ids,
            'audio_ids': full_audio_ids,
            'text_str': full_text_str,
            'audio_str': full_audio_str,
        }
    
    def get_results(self):
        accumulated = self.get_accumulated_results()
        return accumulated['text_ids'], accumulated['audio_ids']

def remove_generate_text_special_token(text):
    """remove special tokens"""
    text = re.sub(r'^(<\|audio_bos\|>|<\|sil\|>)+', '', text)
    text = re.sub(r'(<\|audio_eos\|>|<\|sil\|>|<\|im_end\|>)+$', '', text)
    return text
