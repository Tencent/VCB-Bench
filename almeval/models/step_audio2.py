
import torch
from .base import BaseModel
from .stepaudio2.stepaudio2 import StepAudio2 as StepAudio2Model
from .stepaudio2.stepaudio2 import StepAudio2Base as StepAudio2BaseModel
from .stepaudio2.token2wav import Token2wav
from .stepaudio2.utils import padding_mels
from transformers import  GenerationConfig
import s3tokenizer

class StepAudio2Base(BaseModel):
    NAME = 'StepAudio2-Base'

    def __init__(self, model_path='stepfun-ai/Step-Audio-2-mini-Base', **kwargs):
        assert model_path is not None
        self.model = StepAudio2BaseModel(model_path)
        # self.token2wav = Token2wav(f'{model_path}/token2wav')

        super().__init__()

    def generate_inner(self, msg: dict):
        # import pdb; pdb.set_trace()
        if msg['meta']['task'] == 'Pretrain':
            # speech to text
            messages = [{"type": "audio", "audio": msg['audio'][0]},msg['answer']]
            s2t_pos_loss=self.forward_inner(messages)
            messages = [{"type": "audio", "audio": msg['audio'][0]},msg['wrong_answer']]
            s2t_neg_loss=self.forward_inner(messages)
            # speech to speech
            messages = [{"type": "audio", "audio": msg['audio'][0]},{"type": "audio", "audio": msg["answer_audio_path"]}]
            s2s_pos_loss=self.forward_inner(messages)
            messages = [{"type": "audio", "audio": msg['audio'][0]},{"type": "audio", "audio": msg["wrong_answer_audio_path"]}]
            s2s_neg_loss=self.forward_inner(messages)
            
            res={
                's2t_pos_loss': s2t_pos_loss,
                's2t_neg_loss': s2t_neg_loss,
                's2s_pos_loss': s2s_pos_loss,
                's2s_neg_loss': s2s_neg_loss,
            }

            return "", res, None
        else:
            raise NotImplementedError(f'Not implemented task {msg["meta"]["task"]}')
    
    def forward_inner(self,messages):
        messages, mels = self.model.apply_chat_template(messages)

        # Tokenize prompts
        prompt_ids = []
        for msg in messages:
            if isinstance(msg, str):
                prompt_ids.append(self.model.llm_tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"])
            elif isinstance(msg, list):
                prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported content type: {type(msg)}")
        prompt_ids = torch.cat(prompt_ids, dim=-1).cuda()
        attention_mask = torch.ones_like(prompt_ids)

        #mels = None if len(mels) == 0 else torch.stack(mels).cuda()
        #mel_lengths = None if mels is None else torch.tensor([mel.shape[1] - 2 for mel in mels], dtype=torch.int32, device='cuda')
        if len(mels)==0:
            mels = None
            mel_lengths = None
        else:
            mels, mel_lengths = padding_mels(mels)
            mels = mels.cuda()
            mel_lengths = mel_lengths.cuda()

        generate_inputs = {
            "input_ids": prompt_ids,
            "wavs": mels,
            "wav_lens": mel_lengths,
            "attention_mask":attention_mask
        }

        ret = self.model.llm(**generate_inputs)
        loss, _=self.mask_loss_ppl(ret.logits, prompt_ids)
        return loss

class StepAudio2(BaseModel):
    NAME = 'StepAudio2'

    def __init__(self, model_path='stepfun-ai/Step-Audio-2-mini', **kwargs):
        assert model_path is not None
        self.model = StepAudio2Model(model_path)
        self.token2wav = Token2wav(f'{model_path}/token2wav')

        super().__init__()

    def get_prompt(self, msg: dict):
        if msg['meta']['interactive'] == 'Audio-analysis':
            # https://github.com/stepfun-ai/Step-Audio2/blob/main/examples.py#L150
            user_prompt = "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."
            return user_prompt
        else:
            user_prompt = "You are a helpful assistant." 
            if msg['text'] is not None and msg['text'] != '':
                return  f"{user_prompt} {msg['text']}"
            return user_prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        task_type=msg['task_type']
        # if len(audio) == 1:
            # audio = audio[0]

        prompt = self.get_prompt(msg)
        messages = [{"role": "system", "content": prompt}]
        #import pdb;pdb.set_trace()
        if msg['meta']['interactive'] == 'Audio-analysis':
            if len(audio) > 1:
                raise NotImplementedError
            audio = audio[0]
            text = msg["text"]
            messages.append({"role": "human", "content": [{"type": "audio", "audio": audio},{"type": "text", "text": text}]})
        elif msg['meta']['interactive'] == 'Audio-QA':
            for i in range(len(audio)):
                if len(audio)>1:
                    if msg['meta']['context'] is None:
                        raise ValueError("Assistant's text must be provided for multi-turn dialog task.")
                    text=msg['meta']['context']
                if i%2==0:
                    messages.append({"role": "human", "content": [{"type": "audio", "audio": audio[i]}]})
                else:
                    # https://github.com/XiaomiMiMo/MiMo-Audio-Eval/blob/main/slm_eval/models/step_audio2.py#L144
                    # 这么写history多了会乱码
                    # assist_audio_path = audio[i]
                    # _audio = s3tokenizer.load_audio(assist_audio_path, sr=16000)  # [T]
                    # mels = s3tokenizer.log_mel_spectrogram(_audio)
                    # mels, mels_lens = s3tokenizer.padding([mels])
                    # speech_tokens, _ = self.token2wav.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
                    # messages.append({"role": "assistant", "content":[{"type": "text", "text":"<tts_start>"}, {"type":"token", "token": speech_tokens.cpu().tolist()[0]}]})
                    
                    messages.append({"role": "assistant", "content": [{"type": "audio", "audio": audio[i]},{"type": "text", "text": text[i]}]})
        else:
            raise NotImplementedError
        # import pdb;pdb.set_trace()
        if task_type=="audio2audio":
            # aqaa
            # https://github.com/stepfun-ai/Step-Audio2/blob/main/examples.py#L71
            messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})
            tokens, text, audio = self.model(messages, max_tokens=2048, temperature=0.7, do_sample=True)
            #print(tokens)
            audio = [x for x in audio if x < 6561] # remove audio padding
            sample_rate,wav = self.token2wav(audio, prompt_wav='almeval/models/stepaudio2/assets/default_female.wav')
            # import pdb;pdb.set_trace()
            # with open('output.wav', 'wb') as f:
                # f.write(audio)

            return prompt, text, (sample_rate,wav.detach().cpu())
        else:
            # aqta
            # https://github.com/stepfun-ai/Step-Audio2/blob/main/examples.py#L50
            messages.append({"role": "assistant", "content": None})
            _, text, _ = self.model(messages, max_new_tokens=256, temperature=0.5, do_sample=True)
            # import pdb;pdb.set_trace()
            return prompt, text, None

