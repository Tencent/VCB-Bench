import sys
sys.path.insert(0, 'almeval/models/kimi_audio') #noqa
import torch
from kimia_infer.api.kimia import KimiAudio as KimiAudio_hf
from kimia_infer.api.prompt_manager import KimiAPromptManager
from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.model import KimiAudioModel
from transformers import  AutoTokenizer,AutoConfig
from .base import BaseModel
from transformers import AutoModelForCausalLM
from torch.cuda.amp import autocast
import os
import tempfile
from pydub import AudioSegment
class KimiAudioBase(BaseModel):
    NAME = 'Kimi-Audio-Base'

    def __init__(self, model_path='moonshotai/Kimi-Audio-7B', **kwargs):
        assert model_path is not None
        self.model = KimiAudioModel.init_from_pretrained(model_path,model_load_kwargs={}).to("cuda")

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model_max_length = 512
        
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        self.prompt_manager = KimiAPromptManager(
                model_path=model_path, kimia_token_offset=self.model_config.kimia_token_offset, kimia_text_audiodelaytokens=self.model_config.kimia_mimo_audiodelaytokens
            )

        self.data_module=LazySupervisedDataset(
            [], 
            whisper_model=self.model.whisper_model, 
            text_tokenizer=self.text_tokenizer, 
            max_len=self.model_max_length, 
            kimia_token_offset=self.model.config.kimia_token_offset)

        super().__init__()

    def get_prompt(self, msg: dict):
        return msg['text']

    def generate_inner(self, msg: dict):
        # import pdb;pdb.set_trace()
        if msg['meta']['task'] == 'Pretrain':
            # speech to text
            conversation = [
                {'role': 'assistant', 'message_type': 'audio', 'content': msg["audio"][0]},
                {'role': 'assistant', 'message_type': 'text', 'content': msg["answer"]},    
            ]
            s2t_pos_loss=self.forward_inner(conversation)
            conversation = [
                {'role': 'assistant', 'message_type': 'audio', 'content': msg["audio"][0]},
                {'role': 'assistant', 'message_type': 'text', 'content': msg["wrong_answer"]},    
            ]
            s2t_neg_loss=self.forward_inner(conversation)
            # speech to speech
            # import pdb;pdb.set_trace()
            merge_audio_path=self.merge_audio([msg["audio"][0],msg["answer_audio_path"]])
            conversation = [
                {'role': 'assistant', 'message_type': 'audio', 'content': merge_audio_path},
                # {'role': 'assistant', 'message_type': 'audio', 'content': msg["answer_audio_path"], 'audio_tokens': self.prompt_manager._tokenize_audio(msg['answer_audio_path'])},    
            ]
            s2s_pos_loss=self.forward_inner(conversation)
            os.unlink(merge_audio_path)
            merge_audio_path=self.merge_audio([msg["audio"][0],msg["wrong_answer_audio_path"]])
            conversation = [
                {'role': 'assistant', 'message_type': 'audio', 'content': merge_audio_path},
                # {'role': 'assistant', 'message_type': 'audio', 'content': msg["wrong_answer_audio_path"], 'audio_tokens': self.prompt_manager._tokenize_audio(msg['wrong_answer_audio_path'])},    
            ]
            s2s_neg_loss=self.forward_inner(conversation)
            os.unlink(merge_audio_path)
            res={
                's2t_pos_loss': s2t_pos_loss,
                's2t_neg_loss': s2t_neg_loss,
                's2s_pos_loss': s2s_pos_loss,
                's2s_neg_loss': s2s_neg_loss,
            }

            return "", res, None        
        else:
            raise NotImplementedError(f'Not implemented task {msg["meta"]["task"]}')
    
    # https://github.com/MoonshotAI/Kimi-Audio/tree/master/finetune_codes
    def forward_inner(self,conversation):
        tokenized_conversation = self.prompt_manager.get_prompt(conversation, output_type="text", add_assistant_start_msg=False, numpy_audio_features=True)

        audio_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, text_token_loss_mask = tokenized_conversation.to_tensor()

        audio_input_ids = audio_input_ids.to(self.model.device)
        text_input_ids = text_input_ids.to(self.model.device)
        is_continuous_mask = is_continuous_mask.to(self.model.device)
        audio_token_loss_mask = audio_token_loss_mask.to(self.model.device)
        text_token_loss_mask = text_token_loss_mask.to(self.model.device)

        audio_features = tokenized_conversation.continuous_feature

        audio_labels = torch.cat((audio_input_ids[:, 1:], audio_input_ids.new_full((1, 1), self.data_module.pad_token)), dim=1)
        text_labels = torch.cat((text_input_ids[:, 1:], text_input_ids.new_full((1, 1), self.data_module.pad_token)), dim=1)
        audio_loss_mask = torch.cat((audio_token_loss_mask[:, 1:], audio_token_loss_mask.new_full((1, 1), False)), dim=1)
        text_loss_mask = torch.cat((text_token_loss_mask[:, 1:], text_token_loss_mask.new_full((1, 1), False)), dim=1)

        ret = dict(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=audio_features,
            is_continuous_mask=is_continuous_mask,
            labels=(
                audio_labels,
                text_labels,
                audio_loss_mask,
                text_loss_mask,
            ),
        )
        with autocast(dtype=torch.bfloat16):
            ret=self.model(**ret)
        # import pdb;pdb.set_trace()
        loss=self.compute_loss(ret,[audio_labels, text_labels, audio_loss_mask, text_loss_mask ])
        return loss

    def compute_loss(self, outputs, labels, num_items_in_batch=None):

        audio_logits, text_logits = outputs.logits

        audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
        assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")

        audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
        text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")


        audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
        text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
        loss = audio_loss + text_loss
        return loss.item()
    
    def merge_audio(self,audio_paths):
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        merged_audio_path = temp_file.name
        temp_file.close()

        merged_audio = None
        for audio_path in audio_paths:
            audio_segment = AudioSegment.from_file(audio_path)
            merged_audio = audio_segment if merged_audio is None else merged_audio + audio_segment

        merged_audio.export(merged_audio_path, format="wav")
        return merged_audio_path

class KimiAudio(BaseModel):
    NAME = 'Kimi-Audio'

    def __init__(self, model_path='moonshotai/Kimi-Audio-7B-Instruct', **kwargs):
        assert model_path is not None
        self.model = KimiAudio_hf(
            model_path=model_path, load_detokenizer=True)

        self.sampling_params = {
            'audio_temperature': 0.8,
            'audio_top_k': 10,
            'text_temperature': 0.0,
            'text_top_k': 5,
            'audio_repetition_penalty': 1.0,
            'audio_repetition_window_size': 64,
            'text_repetition_penalty': 1.1,
            'text_repetition_window_size': 16,
            'max_new_tokens': -1,  # TODO: set it
        }
        super().__init__()

        self.sample_rate=24000
    def get_prompt(self, msg: dict):
        return msg['text']

    def generate_inner(self, msg: dict):

        audio = msg['audio']
        task_type=msg['task_type']

        # if len(audio) == 1:
            # audio = audio[0]
        # else:
            # raise NotImplementedError(
                # f'Audio length {len(audio)} not supported')
        if len(audio)>1:
            if msg['meta']['context'] is None:
                raise ValueError("Assistant's text must be provided for multi-turn dialog task.")
            text=msg['meta']['context']

        audiogen_flag=True if task_type=="audio2audio" else False

        prompt = self.get_prompt(msg)

        messages = []

        if prompt is not None and prompt.strip() != '':
            messages.append(
                {'role': 'user', 'message_type': 'text', 'content': prompt})
        # import pdb;pdb.set_trace()
        for i in range(len(audio)):
            if i%2==0:
                messages.append(
                    {'role': 'user', 'message_type': 'audio', 'content': audio[i]})
            else:
                #messages.append(
                #    {'role': 'assistant', 'message_type': 'audio', 'content': audio[i]})
                messages.append(
                    {'role': 'assistant', 'message_type': 'text', 'content': text[i]})

        #import pdb;pdb.set_trace()
        if not audiogen_flag:
            _, text = self.model.generate(
                messages, **self.sampling_params, output_type='text')
            return prompt,text,None
        else:
            wav_output, text = self.model.generate(
                messages, **self.sampling_params, output_type='both')
            # print(wav_output.shape)
        return prompt, text,(self.sample_rate,wav_output.detach().cpu())
