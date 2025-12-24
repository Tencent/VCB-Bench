import random
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from ..utils.misc import print_once
from .base import BaseModel


class Qwen3_Omni(BaseModel):
    NAME = 'Qwen3-Omni-Instruct'

    def __init__(self, model_path='Qwen/Qwen3-Omni-30B-A3B-Instruct', **kwargs):
        assert model_path is not None
        self.model_path = model_path

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

        self.sample_rate=24000
        random.seed(0)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        # according to https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
        meta = msg['meta']
        if meta['task'] == 'ASR':
            assert 'lang' in meta
            lang = meta['lang']
            if lang == 'zh':
                prompt = '请将这段中文语音转换为纯文本，去掉标点符号。'
            elif lang == 'en':
                prompt = 'Transcribe the English audio into text without any punctuation marks.'
            else:
                raise NotImplementedError
        elif meta['dataset_name'] == 'vocalsound':
            # from https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
            prompt = 'Classify the given human vocal sound in English.'
        elif meta['dataset_name'] == 'meld':
            # from: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
            prompt = 'Recognize the emotion with keywords in English.'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]}.'
        else:
            prompt = msg['text']
        return prompt

    def get_system_prompt(self, msg: dict, audiogen_flag: bool):
        meta = msg['meta']
        if meta is None:
            return ''
        # from: https://github.com/QwenLM/Qwen2.5-Omni/blob/6c1784249f8aa498a0893ec442e20557c2fa5773/web_demo.py#L41C29-L41C192
        system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
        if meta['task'] == 'ASR':
            # from: https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
            system_prompt = 'You are a speech recognition model.'
        elif meta['dataset_name'] in ['vocalsound', 'Nonspeech7k']:
            system_prompt = 'You are a vocal sound classification model.'
        elif meta['dataset_name'] == 'meld':
            system_prompt = 'You are a speech emotion recognition model.'
        elif meta['interactive'] == 'Audio-QA' or meta['audio_type'] == 'AudioEvent':
            system_prompt = 'You are a helpful assistant.'
        return system_prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        task_type=msg['task_type']
        # if len(audio) == 1:
        #     audio = audio[0]
        audiogen_flag=True if task_type=="audio2audio" else False
        task_prompt = self.get_prompt(msg)
        system_prompt = self.get_system_prompt(msg, audiogen_flag=audiogen_flag)

        if msg['meta']['interactive'] == 'Audio-analysis':
            if len(audio) > 1:
                raise NotImplementedError
            audio = audio[0]
            messages = [
                {'role': 'system','content': [{'type': 'text','text': system_prompt}]},
                {'role': 'user','content': [{'type': 'audio','audio': audio}, {'type': 'text','text': task_prompt}]},
            ]
        elif msg['meta']['interactive'] == 'Audio-QA':
            if len(audio)>1:
                if msg['meta']['context'] is None:
                    raise ValueError("Only a2ta supported: Text must be provided for multi-turn dialog task.")
                text=msg['meta']['context']
            messages = [{'role': 'system','content': [{'type': 'text','text': system_prompt}]}]
            for i in range(len(audio)):
                if i%2==0:
                    messages.append({'role': 'user','content': [{'type': 'audio','audio': audio[i]}]})
                else:
                    # messages.append({'role': 'assistant','content': [{'type': 'audio','audio': audio[i]},{'type': 'text','text': text[i]}]})
                    messages.append({'role': 'assistant','content': [{'type': 'text','text': text[i]}]})
        else:
            raise NotImplementedError
        # only for dump
        # import pdb; pdb.set_trace()
        prompt = system_prompt + '\n' + task_prompt
        print_once(f'Prompt: {prompt}')

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True)
        # assert audio is not None

        inputs = self.processor(text=text,
                                audio=audios,
                                images=images,
                                videos=videos,
                                return_tensors='pt',
                                padding=True, use_audio_in_video=True)

        inputs = inputs.to('cuda').to(self.model.dtype)
        # import pdb; pdb.set_trace()
        if not audiogen_flag:
            text_ids, audio = self.model.generate(**inputs, 
                                            speaker="Ethan", 
                                            thinker_return_dict_in_generate=True,
                                            use_audio_in_video=True,
                                            return_audio=False)
        else:
            text_ids, audio = self.model.generate(**inputs, 
                                            speaker="Ethan", 
                                            thinker_return_dict_in_generate=True,
                                            use_audio_in_video=True,
                                            return_audio=True)

        text = self.processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        text = text[0]
        if not audiogen_flag:
            return prompt, text, None
        else:
            return prompt, text, (self.sample_rate,audio.squeeze(0).detach().cpu())
