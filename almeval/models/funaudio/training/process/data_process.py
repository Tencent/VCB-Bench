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

import datasets
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.constant import DEFAULT_S2M_PROMPT, AUDIO_TEMPLATE, AUDIO_PAD_TOKEN, TOKEN_FPS, AUDIO_BOS_TOKEN
import soundfile as sf
import json
from utils.cosyvoice_tokenizer import extract_speech_token, get_audio_tokenizer
import torch
from multiprocess import set_start_method
import argparse

# 使用全局字典存储每个进程的模型和处理器
process_globals = {}

def convert_format(example, rank, audio_dir, delay_token_nums=10):
    global process_globals
    num_gpus = torch.cuda.device_count()
    
    # 每个进程只加载一次 ort_session
    if "ort_session" not in process_globals:
        onnx_path = "../pretrained_models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx"
        # 根据 rank 分配到不同的 GPU，使用取模运算实现轮询分配
        num_gpus = torch.cuda.device_count()
        device_id = rank % num_gpus
        print(f"Loading audio tokenizer for process rank {rank} on GPU {device_id}")
        process_globals["ort_session"] = get_audio_tokenizer(onnx_path, device_id=device_id)
    
    ort_session = process_globals["ort_session"]
    
    messages = [
        {"role": "user", "content": AUDIO_TEMPLATE},
        {"role": "assistant", "content": AUDIO_TEMPLATE}
    ]

    # Save input audio to local
    audio_data = example['input_audio']
    input_audio_path = f"{audio_dir}/{audio_data['path']}"
    sf.write(input_audio_path, audio_data['array'], audio_data['sampling_rate'])

    # Save output audio to local
    audio_data = example['output_audio']
    output_audio_path = f"{audio_dir}/{audio_data['path']}"
    sf.write(output_audio_path, audio_data['array'], audio_data['sampling_rate'])

    assistant_tokens = extract_speech_token(ort_session, output_audio_path)

    audios = [
        {
            "path": os.path.realpath(input_audio_path),
            "text": "",
            "token": AUDIO_PAD_TOKEN * int(audio_data['array'].shape[0] / audio_data['sampling_rate'] * TOKEN_FPS),
            "ref_path": "",
            "ref_text": example['speech_input'],
        },
        {
            "path": "",
            "text": example['output'],
            "token": AUDIO_BOS_TOKEN * delay_token_nums + ''.join([f'[AU{token:04d}]' for token in assistant_tokens]),
            "ref_path": os.path.realpath(output_audio_path),
            "ref_text": "",
        }
    ]
    audios = [json.dumps(audio, ensure_ascii=False, sort_keys=True) for audio in audios]

    return {"system": DEFAULT_S2M_PROMPT, "messages": messages, "audio": audios}

def main():
    parser = argparse.ArgumentParser(description="Process audio dataset for training")
    parser.add_argument("--debug", action="store_true", help="Debug mode, only process limited samples")
    parser.add_argument("--datapath", type=str, default="datasets/spoken-alpaca-gpt4", help="Path to dataset")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers (default: torch.cuda.device_count() * 4)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process in debug mode")
    parser.add_argument("--delay_token_nums", type=int, default=10, help="Number of delay tokens")
    args = parser.parse_args()
    
    set_start_method("spawn")
    
    num_workers = args.num_workers if args.num_workers is not None else torch.cuda.device_count() * 4

    datapath = args.datapath
    audio_dir = os.path.join(datapath, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    dataset = datasets.load_from_disk(datapath)
    if args.debug:
        dataset = dataset.select(range(args.num_samples))
        
    dataset = dataset.map(
        convert_format, 
        with_rank=True, 
        remove_columns=dataset.column_names,
        num_proc=num_workers, 
        fn_kwargs={"audio_dir": audio_dir, "delay_token_nums": args.delay_token_nums}, 
        desc="Converting format of dataset",
    )
    dataset = dataset.remove_columns([ dn for dn in dataset.column_names if dn not in ["system", "messages", "audio"]])
    
    output_filename = "train.jsonl" if not args.debug else f"train_{args.num_samples//1000}k.jsonl"
    output_path = os.path.join(datapath, output_filename)
    dataset.to_json(output_path, orient="records", lines=True)
    print(f"Saved to: {output_path}")
    
if __name__ == "__main__":
    main()
