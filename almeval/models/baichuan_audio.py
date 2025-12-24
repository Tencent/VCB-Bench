import json
import ujson
import re
import types
import warnings
import sys
import os
import numpy as np
import math

import torch
from torch.nn import functional as F
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.misc import print_once
from .base import BaseModel
from .patch import patch_baichuan_load_audio_waveform
from .baichuan.generation import GenerationAudioTokens, decode_save_concat,decode_wave_vocoder

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.set_num_threads(1)


class BaichuanAudioBase(BaseModel):
    NAME = 'Baichuan-Audio'

    def __init__(self, model_path='baichuan-inc/Baichuan-Audio-Base', **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, model_max_length=128000)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.model.config.use_cache = True
        self.model.bind_processor(
            self.tokenizer, training=False, relative_path='/')

        audio_processor = self.model.processor.audio_processor
        audio_processor.load_audio_waveform = types.MethodType(
            patch_baichuan_load_audio_waveform, audio_processor)

        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        if meta['task'] == 'ASR':
            # from: https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/base_asr_demo.py#L84C19-L84C27
            prompt = '将语音转录为文本:'
        # to invoke basemodel continuous output
        elif meta['interactive'] == 'Audio-QA':
            prompt = '对音频中的问题，你的回答是:'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]} 你的回答是: '
        else:
            prompt = msg['text']
            # to invoke basemodel continuous output
            end_punctuation = ['.', '?', '!', '。', '？', '！']
            if prompt.endswith(tuple(end_punctuation)):
                prompt = prompt + ' Your answer to this question is:'
            else:
                prompt = prompt + ' . ' + 'Your answer to this question is:'
        return prompt

    def generate_inner(self, msg: dict):
        # import pdb;pdb.set_trace()
        if msg['meta']['task'] == 'Pretrain':      
            # speech to text
            prompt='下面是一段语音和文本交错的内容：<audio_start_baichuan>{{\"path\": \"{0}\"}}<audio_end_baichuan><trainable_start>{1}<trainable_end>'
            s2t_pos_loss,_=self.forward_inner(prompt.format(msg["audio"][0], msg["answer"]))
            s2t_neg_loss,_=self.forward_inner(prompt.format(msg["audio"][0], msg["wrong_answer"]))
            # speech to speech
            prompt='下面是一段语音和语音交错的内容：<audio_start_baichuan>{{\"path\": \"{0}\"}}<audio_end_baichuan><audio_start_baichuan>{{\"path\": \"{1}\"}}<audio_end_baichuan>'
            s2s_pos_loss,_=self.forward_inner(prompt.format(msg["audio"][0], msg["answer_audio_path"]))
            s2s_neg_loss,_=self.forward_inner(prompt.format(msg["audio"][0], msg["wrong_answer_audio_path"]))
            res={
                's2t_pos_loss': s2t_pos_loss,
                's2t_neg_loss': s2t_neg_loss,
                's2s_pos_loss': s2s_pos_loss,
                's2s_neg_loss': s2s_neg_loss,
            }   
            return prompt,res,None
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]
        audio_tokens_all = []
        prompt = self.get_prompt(msg)
        audio_tokens = self.audio_start_token + \
            json.dumps({'path': audio}) + self.audio_end_token
        audio_tokens_all.append(audio_tokens)

        print_once(f'Prompt: {prompt}')
        prompt = prompt + ''.join(audio_tokens_all)

        ret = self.model.processor([prompt])
        ret_audios = ret.audios.cuda() if ret.audios is not None else None
        ret_encoder_length = ret.encoder_length.cuda(
        ) if ret.encoder_length is not None else None
        ret_bridge_length = ret.bridge_length.cuda(
        ) if ret.bridge_length is not None else None
        predicted_ids = self.model.generate(input_ids=ret.input_ids.cuda(),
                                            attention_mask=ret.attention_mask.cuda(),
                                            labels=None,
                                            audios=ret_audios,
                                            encoder_length=ret_encoder_length,
                                            bridge_length=ret_bridge_length,
                                            max_new_tokens=700,
                                            num_beams=1,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            repetition_penalty=1.3)
        generated = self.tokenizer.batch_decode(
            predicted_ids[:, ret.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return prompt, generated, None

    # https://github.com/undobug/S2SBench/blob/main/s2t/s2t_infer_ppl.py
    def forward_inner(self,input_strings):
        batch_data = self.model.processor([input_strings])
        batch_position_ids = self.create_position_ids_from_input_ids_left_padded(batch_data.input_ids,batch_data.attention_mask)

        input_ids=batch_data['input_ids']

        ret = self.model(
            input_ids=batch_data['input_ids'].cuda(),
            attention_mask=batch_data.attention_mask.cuda(),
            position_ids = batch_position_ids.cuda(),
            labels=None, 
            audios=batch_data['audios'].cuda(),
            encoder_length=batch_data['encoder_length'].cuda(),
            bridge_length=batch_data['bridge_length'].cuda()
        )
        loss,ppl = self.mask_loss_ppl(ret.logits,input_ids)
        return loss,ppl

    def create_position_ids_from_input_ids_left_padded(self,input_ids, attention_mask, past_key_values_length=0):

        seq_lengths = attention_mask.sum(dim=1)
        position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
        for i in range(input_ids.size(0)):
            actual_seq_length = seq_lengths[i].item()
            if actual_seq_length > 0:
                position_ids[i, -actual_seq_length:] = torch.arange(
                    past_key_values_length, past_key_values_length + actual_seq_length, 
                    dtype=torch.long, device=input_ids.device
                )

        return position_ids



# https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/s2s_gradio_demo_cosy_multiturn.py
class BaichuanAudioChat(BaichuanAudioBase):
    """Chat model for Audio-QA only, no text prompt given.
    """
    role_prefix = {
        'system': '<B_SYS>',
        'user': '<C_Q>',
        'assistant': '<C_A>',
        'audiogen': '<audiotext_start_baichuan>'
    }
    special_token_partten = re.compile(
        r'<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>')
    sample_rate = 24000
    NAME = 'Baichuan-Audio-Chat'

    def __init__(self, model_path='baichuan-inc/Baichuan-Audio-Instruct', **kwargs):
        super().__init__(model_path, **kwargs)
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_end_token_id)

        sys.path.append(os.path.join('./almeval/models/baichuan/third_party/cosy24k_vocoder'))
        from cosy24k_vocoder import Cosy24kVocoder
        # vocoder = Cosy24kVocoder.from_pretrained(os.path.join(args.vocoder, "hift.pt"))
        self.vocoder = Cosy24kVocoder.from_pretrained("./almeval/models/baichuan/third_party/cosy24k_vocoder/hift.pt")
        self.vocoder = self.vocoder.cuda()
        self.wave_concat_overlap = int(self.sample_rate * 0.01)

    def wave_concat(self,wave_list, start, overlap=400):
        new_wave_list = []
        cur = start
        for wave in wave_list[start:]:
            if (
                cur - 1 >= 0
                and wave_list[cur - 1].shape[1] > overlap
                and wave.shape[1] > overlap
            ):
                new_wave_list.append(
                    (
                        wave_list[cur - 1][:, -overlap:]
                        * torch.linspace(
                            1.0, 0.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                        + wave[:, :overlap]
                        * torch.linspace(
                            0.0, 1.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                    )
                )
            new_wave_list.append(wave)
            cur += 1
        return torch.cat(new_wave_list, dim=1)

    def save_local(self,wave, local_path):
        torchaudio.save(local_path, torch.cat(wave, dim=0).cpu(), self.sample_rate)
        return self.audiogen_start_token + ujson.dumps({'path': local_path}, ensure_ascii=False) + self.audiogen_end_token

    def generate_text_step(self,pret, plen, kv_cache_flag, audiogen_flag):
        if not kv_cache_flag:
            textret = self.model.generate(
                    pret.input_ids.cuda(),
                    attention_mask=pret.attention_mask.cuda(), 
                    audios = pret.audios.cuda() if pret.audios is not None else None,
                    encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
                    bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
                    tokenizer=self.tokenizer,
                    max_new_tokens=50 if audiogen_flag else 1024,
                    stop_strings=[self.audiogen_start_token, '<|endoftext|>'] if audiogen_flag else ['<|endoftext|>'],
                    do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
                )
        else:
            textret =self. model.generate(
                    pret.sequences,
                    attention_mask=torch.ones_like(pret.sequences),
                    tokenizer=self.tokenizer,
                    past_key_values=(pret.past_key_values),
                    stop_strings = [self.audiogen_start_token,',','!','?','，','。','！','？','. '],
                    max_new_tokens=50, do_sample=True, temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.05, return_dict_in_generate=True,
                )
        newtext = self.tokenizer.decode(textret.sequences[0, plen:])
        return textret, newtext

    def generate_audio_step(self,pret):
        audioret = GenerationAudioTokens.generate(
                    self.model,
                    pret.sequences,
                    attention_mask=torch.ones_like(pret.sequences),
                    past_key_values=(pret.past_key_values if pret.past_key_values is not None else None),
                    max_new_tokens=500,
                    do_sample=True, temperature=0.5, top_k=5, top_p=0.85, repetition_penalty=1.3, return_dict_in_generate=True,
        )
        wave_segment = decode_wave_vocoder(audioret.audios_sequences.clone(), self.vocoder, self.model)
        return audioret, wave_segment


    def preprocess_messages(self,messages, audiogen_flag):
        text = ""
        for i, msg in enumerate(messages):
            if audiogen_flag and msg["role"] == "assistant":
                text += self.role_prefix['audiogen']
            text += self.role_prefix[msg['role']]
            text += msg['content']
        if audiogen_flag:
            text += self.role_prefix['audiogen']
        text += self.role_prefix["assistant"]
        return text

    def get_prompt(self, msg: dict):
        # according to https://github.com/baichuan-inc/Baichuan-Audio/blob/main/web_demo/base_asr_demo.py
        meta = msg['meta']
        if meta['task'] == 'ASR':
            # from: https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/base_asr_demo.py#L84C19-L84C27
            prompt = '将语音转录为文本。'

        elif meta['interactive'] == 'Audio-QA':
            # from: https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/s2s_gradio_demo_cosy_multiturn.py#L309
            prompt = '请用【邻家女声】这个声音回答问题。'

        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]}'
        else:
            prompt = msg['text']
        return prompt

    def get_audio_tokens(self, audio):
        audio_tokens_all = []
        audio_tokens = self.audio_start_token + \
            json.dumps({'path': audio}) + self.audio_end_token
        audio_tokens_all.append(audio_tokens)
        prompt = ''.join(audio_tokens_all)
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        # import pdb;pdb.set_trace()
        # if len(audio) == 1:
        #     audio = audio[0]
        task_type=msg['task_type']
        audiogen_flag=True if task_type=="audio2audio" else False

        sys_prompt = self.get_prompt(msg)

        msgs = [{'role': 'system','content': sys_prompt}]

        for i in range(len(audio)):
            if len(audio)>1:
                if msg['meta']['context'] is None:
                    raise ValueError("Assistant's text must be provided for multi-turn dialog task.")
                text=msg['meta']['context']
            if i%2 == 0:
                msgs.append({'role': 'user', 'content': self.get_audio_tokens(audio[i])})
            else:
                # msgs.append({'role': 'assistant', 'content': self.get_audio_tokens(audio[i])})
                msgs.append({'role': 'assistant', 'content': text[i]})
        
        message = self.preprocess_messages(msgs,audiogen_flag)
        result= self.generate_response(message,audiogen_flag) # show_text,real_text (sr,audio)
        #import pdb;pdb.set_trace()
        return sys_prompt, result[1], result[2] 
    
    def generate_response(self, content, audiogen_flag):
        pret = self.model.processor([content])
        plen = pret.input_ids.shape[1]
        ret, text_segment = self.generate_text_step(pret, plen, False, audiogen_flag)
        wave_list = []
        full_text = re.sub(self.special_token_partten, '', text_segment)
        show_text = re.sub(self.special_token_partten, '', text_segment)
        text_list=[text_segment]
        # import pdb; pdb.set_trace()
        if audiogen_flag:
            start = 0
            for i in range(100): # 100
                m = ret.sequences[0, -1].item()
                if m == self.tokenizer.eos_token_id:
                    if ret.sequences.shape[1] - plen > 1:
                        ret.sequences[0, -1] = (self.model.config.audio_config.audiogen_start_token_id)
                        ret, wave_segment = self.generate_audio_step(ret)
                        wave_list.extend(wave_segment)
                        # full_text += self.save_local(wave_segment, os.path.join(f'round{i}.wav'))
                        show_text += '<audio>'
                        plen = ret.sequences.shape[1]
                        #print("*************")
                    break

                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
                ret, wave_segment = self.generate_audio_step(ret)
                wave_list.extend(wave_segment)
                # full_text += self.save_local(wave_segment, os.path.join(f'round{i}.wav'))
                show_text += '<audio>'

                # if len(wave_list) > max(1, start):
                #     wave = self.wave_concat(wave_list, start, overlap=self.wave_concat_overlap)
                #     start = len(wave_list)

                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_end_token_id
                plen = ret.sequences.shape[1]
                ret, text_segment = self.generate_text_step(ret, plen, True, audiogen_flag)
                text_list.append(text_segment)
                full_text += re.sub(self.special_token_partten, '', text_segment)
                show_text += re.sub(self.special_token_partten, '', text_segment) 
                #print(f"ROUND {i+1}:", text_segment)
            if ret.sequences.shape[1] - plen > 1:
                ret.sequences[0, -1] = (self.model.config.audio_config.audiogen_start_token_id)
                ret, wave_segment = self.generate_audio_step(ret)
                wave_list.extend(wave_segment)
                # full_text += self.save_local(wave_segment, os.path.join(f'round{i}.wav'))
                show_text += '<audio>'
                #print("---------------------")
            # if len(wave_list) > start:
            # assert len(wave_list)==len(text_list)
            new_wav_list=[]
            for i in range(len(wave_list)):
                if text_list[i].strip():
                    new_wav_list.append(wave_list[i])
            wave = self.wave_concat(new_wav_list, start, overlap=self.wave_concat_overlap)

            return show_text, full_text, (self.sample_rate, wave.cpu())
        
        return show_text, full_text, None

    
