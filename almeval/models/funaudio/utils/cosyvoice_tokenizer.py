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

import torchaudio
import numpy as np
import whisper
import onnxruntime
from loguru import logger
import torch

def extract_speech_token(ort_session, wav_path, pool_executor=None):
    def tokenizer(audio_segment):
        feat = whisper.log_mel_spectrogram(audio_segment, n_mels=128)
        speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                              ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[
            0].flatten().tolist()
        return speech_token

    if isinstance(wav_path, str):
        audio, sample_rate = torchaudio.load(wav_path, backend='soundfile')
    else:
        audio = wav_path
        sample_rate = 16000
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    # print(f"audio.shape: {audio.shape}")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    time_step = 0
    audios =[]

    # Step 1: Split audio into 30-second segments
    while time_step * 16000 < audio.shape[1]:
        start = time_step * 16000
        end = min((time_step + 30) * 16000, audio.shape[1])
        audio_segment = audio[:, start:end]
        audios.append(audio_segment)
        time_step += 30
    
    # Step 2: Handle last segment if too short
    if len(audios) > 1 and audios[-1].shape[1] < 16000:  # Less than 1 second
        # Remove last two segments
        last_segment = audios.pop()
        second_last_segment = audios.pop()
        
        # Merge last two segments
        merged_audio = torch.cat([second_last_segment, last_segment], dim=1)
        total_length = merged_audio.shape[1]
        
        # Split merged audio into two equal parts
        split_point = total_length // 2
        first_half = merged_audio[:, :split_point]
        second_half = merged_audio[:, split_point:]
        
        # Add new segments back to list
        audios.append(first_half)
        audios.append(second_half)

    all_speech_tokens = []
    if pool_executor is not None:
        # 提交所有任务
        futures = [
            pool_executor.submit(
                tokenizer,
                item,
            )
            for item in audios
        ]
        for future in futures:
            all_speech_tokens.extend(future.result())
    else:
        for audio_segment in audios:
            speech_token = tokenizer(audio_segment)
            all_speech_tokens.extend(speech_token)
    return all_speech_tokens


def get_audio_tokenizer(onnx_path = "pretrained_models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx", device_id=0):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    # 指定使用哪个 GPU
    providers = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider"
    ]
    ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
    return ort_session
