import librosa
import torch
import torchaudio
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
from .base import BaseModel
from .funaudio.funaudiochat.register import register_funaudiochat
register_funaudiochat()

from .funaudio.utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
from .funaudio.utils.constant import (
    DEFAULT_S2T_PROMPT,
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    DEFAULT_S2M_PROMPT,
    SPOKEN_S2M_PROMPT,
    AUDIO_TEMPLATE,
    FUNCTION_CALLING_PROMPT,
)


class FunAudioChat(BaseModel):
    NAME = 'FunAudioChat'

    def __init__(self, model_path='FunAudioLLM/Fun-Audio-Chat-8B', **kwargs):
        assert model_path is not None
        config = AutoConfig.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map="cuda")
        self.cosyvoice_model = get_audio_detokenizer()

        super().__init__()

    def get_prompt(self, msg: dict):
        if msg["task_type"]=="audio2audio":
            # https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/examples/infer_s2s.py
            return SPOKEN_S2M_PROMPT
        else:
            # https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/examples/infer_s2t.py
            return DEFAULT_S2T_PROMPT

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        task_type=msg['task_type']
        # import pdb;pdb.set_trace()
        if len(audio) > 1:
            raise NotImplementedError

        prompt = self.get_prompt(msg)
        messages = [{"role": "system", "content": prompt}]

        audio = audio[0]
        text = msg["text"]
        if msg['meta']['interactive'] == 'Audio-analysis':
            messages.append({"role": "user", "content": AUDIO_TEMPLATE + "\n" + text})
        else:
            messages.append({"role": "user", "content": AUDIO_TEMPLATE})
        
        audio = [librosa.load(audio, sr=16000)[0]]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(self.model.device)

        if task_type=="audio2audio":
            gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
            gen_kwargs['max_new_tokens'] = 2048
            sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
            sp_gen_kwargs['text_greedy'] = True
            self.model.sp_gen_kwargs.update(sp_gen_kwargs)
            generate_ids, audio_ids = self.model.generate(**inputs, **gen_kwargs)
        else:
            self.model.sp_gen_kwargs.update({
                'text_greedy': True, 
                'disable_speech': True,
            })
            generate_ids, _ = self.model.generate(**inputs)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        generate_text = self.processor.decode(generate_ids[0], skip_special_tokens=True)

        if task_type=="audio2audio":
            # generate_audio = processor.speech_tokenizer.decode(audio_ids[0])
            token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))
            speech = token2wav(self.cosyvoice_model, token_for_cosyvoice, embedding=None, token_hop_len=25 * 30, pre_lookahead_len=3)
            return "", generate_text, (self.cosyvoice_model.sample_rate, speech.detach().cpu())
        else:
            return "", generate_text, None


