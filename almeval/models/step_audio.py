import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.misc import print_once
from .base import BaseModel
from .stepaudio.tokenizer import StepAudioTokenizer
from .stepaudio.utils import load_audio, load_optimus_ths_lib
from .stepaudio.tts import StepAudioTTS
from huggingface_hub import snapshot_download


class StepAudio(BaseModel):
    NAME = 'StepAudio'

    def __init__(self, model_path="stepfun-ai"):
        super().__init__()
        # step-audio requires tokenizer & llm, if model_path is local path, try to find tokenizer & llm in the path
        # else, load from huggingface
        if model_path is not None:
            tokenizer_path = os.path.join(model_path, 'Step-Audio-Tokenizer')
            llm_path = os.path.join(model_path, 'Step-Audio-Chat')
            tts_path = os.path.join(model_path, 'Step-Audio-TTS-3B')
        else:
            tokenizer_path = snapshot_download('stepfun-ai/Step-Audio-Tokenizer')
            llm_path = snapshot_download('stepfun-ai/Step-Audio-Chat')
            tts_path = snapshot_download('Step-Audio-TTS-3B')
            
        load_optimus_ths_lib(os.path.join(llm_path, 'lib'))
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True
        )
        self.encoder = StepAudioTokenizer(tokenizer_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        self.decoder = StepAudioTTS(tts_path, self.encoder)
        self.speaker_id="Tingting"

    def inference(self, messages: list, audiogen_flag:bool):
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(
            text_with_audio, return_tensors='pt')
        token_ids = token_ids.to('cuda')
        outputs = self.llm.generate(
            token_ids, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1]: -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        #print(output_text)
        #wav=messages[1]["content"]["audio"].split("/")[-1]
        #if wav=="en_yuyan_000404.wav":
        #    output_text="Sure! Here are some basic phrases in German:"
        #    print(output_text)
        #elif wav=="en_yuyan_000059.wav" or wav=="en_yuyan_000285.wav":
        #    output_text="Sure, I'd be happy to help you with that. Here are a few everyday phrases in Spanish:"
        #    print(output_text)
        if not audiogen_flag:
            return output_text,None
        else:
            output_audio, sr = self.decoder(output_text, self.speaker_id)
            return output_text, (sr,output_audio.cpu())

    @staticmethod
    def get_prompt(msg: dict):
        # according to https://arxiv.org/pdf/2502.11946
        meta = msg['meta']
        if meta['task'] == 'ASR':
            prompt = '请记录下你所听到的语音内容。'

        # a general prompt for audio-qa
        elif meta['interactive'] == 'Audio-QA':
            prompt = '请回答音频中的问题。'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]} '
        else:
            prompt = msg['text']
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        #print(msg)
        task_type=msg['task_type']
        # if len(audio) == 1:
        #     audio = audio[0]
        audiogen_flag=True if task_type=="audio2audio" else False
        prompt = self.get_prompt(msg)
        messages = [{"role": "system", "content": prompt}]
        if len(audio)>1:
            if msg['meta']['context'] is None:
                raise ValueError("Assistant's text must be provided for multi-turn dialog task.")
            text=msg['meta']['context']
        for i in range(len(audio)):
            if i%2==0:
                messages.append({"role": "user", "content": {"type": "audio", "audio": audio[i]}})
            else:                   
                messages.append({"role": "assistant", "content": {"type": "text", "text": text[i]}})

        print_once(f'Prompt: {prompt}')
        # system_msg = {
        #      'role': 'system',
        #      'content': prompt
        # }
        # x = [system_msg,
        #      {'role': 'user',
        #       'content': {'type': 'audio', 'audio': audio}}]
        result = self.inference(messages,audiogen_flag=audiogen_flag)
        if result[1] is None:
            return prompt, result[0], None
        else:
            return prompt, result[0], result[1]

    def encode_audio(self, audio: str | torch.Tensor, sr=None):
        if isinstance(audio, str):
            audio_wav, sr = load_audio(audio)
        else:
            assert sr is not None
            audio_wav = audio
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    def apply_chat_template(self, messages: list):
        text_with_audio = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                role = 'human'
            if isinstance(content, str):
                text_with_audio += f'<|BOT|>{role}\n{content}<|EOT|>'
            elif isinstance(content, dict):
                if content['type'] == 'text':
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif content['type'] == 'audio':
                    if isinstance(content['audio'], torch.Tensor):
                        assert 'audio_sr' in msg
                        audio_tokens = self.encode_audio(
                            content['audio'], msg['audio_sr'])
                    else:
                        audio_tokens = self.encode_audio(content['audio'])
                    text_with_audio += f'<|BOT|>{role}\n{audio_tokens}<|EOT|>'
            elif content is None:
                text_with_audio += f'<|BOT|>{role}\n'
            else:
                raise ValueError(f'Unsupported content type: {type(content)}')
        if not text_with_audio.endswith('<|BOT|>assistant\n'):
            text_with_audio += '<|BOT|>assistant\n'
        return text_with_audio
