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

import json
import librosa
import torch
import sys
import math
import os
import uuid
import torchaudio
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
from funaudiochat.register import register_funaudiochat
register_funaudiochat()

from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
from utils.constant import (
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    DEFAULT_S2M_PROMPT,
    SPOKEN_S2M_PROMPT,
    AUDIO_TEMPLATE,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def infer_example(model_path, audio_path):
    """
    推理示例函数
    
    Args:
        model_path: 模型路径
        audio_path: 输入音频路径
    """
    # 加载模型和处理器
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)

    # 生成参数
    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs['text_greedy'] = True
    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs['max_new_tokens'] = 2048
    model.sp_gen_kwargs.update(sp_gen_kwargs)

    # 构建audio样例
    audio = [librosa.load(audio_path, sr=16000)[0]]
    
    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
    generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
    generate_audio = processor.speech_tokenizer.decode(audio_ids[0])

    print("generate_text: ", generate_text)
    print("generate_audio_token: ", generate_audio)
    
    token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))

    # 加载CosyVoice detokenizer用于将token转换为wav
    print("Loading CosyVoice detokenizer...")
    cosyvoice_model = get_audio_detokenizer()
    
    # (使用默认的中文女声，你可以根据需要修改）
    print("Converting audio tokens to wav...")
    speech = token2wav(cosyvoice_model, token_for_cosyvoice, embedding=None, token_hop_len=25 * 30, pre_lookahead_len=3)
    
    # 保存wav文件
    output_uuid = str(uuid.uuid4())
    output_path = f'saves/output_audio_{output_uuid}.wav'
    torchaudio.save(output_path, speech.cpu(), cosyvoice_model.sample_rate)
    print(f"Audio saved to: {output_path}")

if __name__ == "__main__":
    model_path = "pretrained_models/Fun-Audio-Chat-8B"
    audio_path = "examples/ck7vv9ag.wav"
    infer_example(model_path, audio_path)