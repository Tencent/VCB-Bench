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

from typing import Optional
import json
import librosa
import torch
import sys
import math
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
from funaudiochat.register import register_funaudiochat
register_funaudiochat()

from utils.constant import (
    DEFAULT_S2T_PROMPT,
    AUDIO_TEMPLATE,
    FUNCTION_CALLING_PROMPT,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def infer_example(model_path, audio_path, instruction: Optional[str] = None):
    """
    推理示例函数（仅生成文本，不生成语音）
    
    Args:
        model_path: 模型路径
        audio_path: 输入音频路径
        instruction: 输入音频指令，可选
    """
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map=device)

    # 生成参数
    model.sp_gen_kwargs.update({
        'text_greedy': True, 
        'disable_speech': True,
    })

    # 构建audio样例
    audio = [librosa.load(audio_path, sr=16000)[0]]
    
    if instruction is None:
        conversation = [
            {"role": "system", "content": DEFAULT_S2T_PROMPT},
            {"role": "user", "content": AUDIO_TEMPLATE},
        ]
    else:
        conversation = [
            {"role": "system", "content": DEFAULT_S2T_PROMPT},
            {"role": "user", "content": AUDIO_TEMPLATE + "\n" + instruction},
        ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
    generate_ids, _ = model.generate(**inputs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)

    print("generate_text: ", generate_text)


def infer_function_calling_example(model_path, audio_path):
    """
    Function Calling 推理示例函数（仅生成文本，不生成语音）

    Args:
        model_path: 模型路径
        audio_path: 输入音频路径
    """
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16,
                                                  device_map=device)

    # 生成参数
    model.sp_gen_kwargs.update({
        'text_greedy': True,
        'disable_speech': True,
    })

    # 构建audio样例
    audio = [librosa.load(audio_path, sr=16000)[0]]

    example_tools = [
        {"type": "function",
         "function": {"name": "get_weather", "description": "查询天气",
                      "parameters": {"type": "object", "properties": {
                          "location": {"type": "string", "description": "地点", "default": "当前位置"},
                          "time": {"type": "string", "description": "时间", "default": "当前时间"}},"required": []}}},
        {"type": "function",
         "function": {"name": "check_battery", "description": "电量查询，例如：现在还剩多少电",
                      "parameters": {"type": "object", "properties": {}, "required": []}}}
    ]

    example_tools_definition = "\n".join([json.dumps(tool_item, ensure_ascii=False) for tool_item in example_tools])
    system_prompt = FUNCTION_CALLING_PROMPT.replace("{tools_definition}", example_tools_definition)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
    generate_ids, _ = model.generate(**inputs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)

    print("generate_text: ", generate_text)


if __name__ == "__main__":
    model_path = "pretrained_models/Fun-Audio-Chat-8B"
    audio_path = "examples/ck7vv9ag.wav"
    infer_example(model_path, audio_path)
