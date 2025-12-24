import torch
import os
import torchaudio
from .base import BaseModel
from .mimoaudio.src.mimo_audio.mimo_audio import MimoAudio as MimoAudioModel

class MimoAudio(BaseModel):
    NAME = 'Mimo-Audio'

    def __init__(self, model_path='XiaomiMiMo', **kwargs):
        assert model_path is not None
        _model_path = os.path.join(model_path,"MiMo-Audio-7B-Instruct")
        tokenizer_path = os.path.join(model_path,"MiMo-Audio-Tokenizer")
        self.model = MimoAudioModel(_model_path, tokenizer_path)
        self.sample_rate = 24000
        
        self.prompt_speech_zh="./almeval/models/mimoaudio/examples/prompt_speech_zh_m.wav"
        self.prompt_speech_en="./almeval/models/mimoaudio/examples/prompt_speech_en_m.wav"
        super().__init__()

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        task_type=msg['task_type']
        if msg["meta"]["lang"]=="en":
            prompt_speech=self.prompt_speech_en
        else:
            prompt_speech=self.prompt_speech_zh

        prompt=""
        # if msg['text'] is not None and msg['text'] != '':
        #     prompt = msg["text"]
        # else:
        #     prompt = ""
        # import pdb;pdb.set_trace()
        if len(audio) == 1:
            # single turn
            audio = audio[0]
            if msg['meta']['interactive'] == 'Audio-analysis':
                if task_type=="audio2audio":
                    raise NotImplementedError
                else:
                    text=msg["text"]
                    # import pdb;pdb.set_trace()
                    text_channel_output = self.model.audio_understanding_sft(audio, text, thinking=False)
                    return prompt, text_channel_output, None
            elif msg['meta']['interactive'] == 'Audio-QA':
                if task_type=="audio2audio":
                    text_channel_output, wav_output = self.model.spoken_dialogue_sft(input_speech=audio, return_audio=True, output_audio_path=None, system_prompt=None, prompt_speech=prompt_speech)
                    # import pdb;pdb.set_trace()
                    text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")
                    return prompt, text_channel_output, (self.sample_rate, wav_output[0].detach().cpu())
                else:
                    text_channel_output = self.model.spoken_dialogue_sft(input_speech=audio, return_audio=False, output_audio_path=None, system_prompt=None, prompt_speech=None)
                    text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")
                    # import pdb;pdb.set_trace()
                    return prompt, text_channel_output, None
            else:
                raise NotImplementedError

        else:
            # multi-turn
            if msg['meta']['context'] is None:
                raise ValueError("Only a2ta supported: Assistant's text must be provided for multi-turn dialog task.")
            text=msg['meta']['context']
            message_list=[]
            for i in range(len(audio)):
                if i%2==0:
                    message_list.append({"role": "user", "content": audio[i]})
                else:
                    message_list.append({"role": "assistant", "content": {"text": text[i], "audio": audio[i]}})
            if task_type=="audio2audio":
                text_channel_output, wav_output= self.model.spoken_dialogue_sft_multiturn(message_list, return_audio=True, output_audio_path=None, system_prompt=None, prompt_speech=prompt_speech)
                text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")
                return prompt, text_channel_output, (self.sample_rate, wav_output[0].detach().cpu())
            else:
                text_channel_output= self.model.spoken_dialogue_sft_multiturn(message_list, return_audio=False, output_audio_path=None, system_prompt=None, prompt_speech=None)
                text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")
                # text_channel_output = self.model.speech2text_dialogue_sft_multiturn(message_list, thinking=False)
                # import pdb;pdb.set_trace()
                return prompt, text_channel_output, None
