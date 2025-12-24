import os
import io
import torchaudio
import base64
from openai import AzureOpenAI
from .base import BaseModel
import time

def audio2base64(audio_path):
  wav_data = open(audio_path, "rb").read()
  encoded_string = base64.b64encode(wav_data).decode('utf-8')
  return encoded_string

class GPT4OAudio(BaseModel):
    NAME = 'GPT4O-Audio'

    def __init__(self, model_path='', **kwargs):
        self.model = AzureOpenAI(
            azure_endpoint = "", 
            api_key="",  
            api_version=""
        )
        self.retry=10
        self.wait=10
        super().__init__()

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        task_type=msg['task_type']
        system_prompt = "Your are a helpful assistant."
        # import pdb;pdb.set_trace()
        if msg['meta']['interactive'] == 'Audio-analysis':
            if len(audio) > 1:
                raise NotImplementedError
            audio = audio[0]
            task_prompt = msg['text']
            message = [
                dict(type='text', text=task_prompt),
                dict(
                    type='input_audio',
                    input_audio=dict(
                        data=audio2base64(audio),
                        format='wav',
                    )
                )
            ]
            input_msgs = [
                    dict(role='system', content=system_prompt),
                    dict(role='user', content=message)
            ]
            _prompt=system_prompt+'\n'+task_prompt
        elif msg['meta']['interactive'] == 'Audio-QA':
            if len(audio) ==1:
                audio=audio[0]
                message = [
                    dict(
                        type='input_audio',
                        input_audio=dict(
                            data=audio2base64(audio),
                            format='wav',
                        )
                    )
                ]
                input_msgs = [
                        dict(role='system', content=system_prompt),
                        dict(role='user', content=message)
                ]
            else:
                if msg['meta']['context'] is None:
                    raise ValueError("Only a2ta supported: Assistant's text must be provided for multi-turn dialog task.")
                text=msg['meta']['context']

                input_msgs = [
                        dict(role='system', content=system_prompt)
                ]
                
                for i in range(len(audio)):
                    if i%2==0:
                        message=[
                            dict(
                                type='input_audio',
                                input_audio=dict(
                                    data=audio2base64(audio[i]),
                                    format='wav',
                                )
                            )
                        ]
                        input_msgs.append(dict(role='user', content=message))
                    else:
                        message=[
                            dict(type='text', text=text[i]),
                            #dict(
                            #    type='input_audio',
                            #    input_audio=dict(
                            #        data=audio2base64(audio[i]),
                            #        format='wav',
                            #    )
                            #)
                        ]
                        input_msgs.append(dict(role='assistant', content=message))
            _prompt=system_prompt
        else:
            raise NotImplementedError
        #import pdb;pdb.set_trace()
        retry=1
        while True:
            try:
                response = self.model.chat.completions.create(
                    model="gpt-4o-audio-preview",
                    modalities=["text", "audio"],
                    audio={"voice": "alloy", "format": "wav"},
                    store=True,
                    messages=input_msgs,
                    max_tokens=2048,
                )

                pred=response.choices[0].message.audio.transcript
                break
            except Exception as e:
                print(f"An error occurred when retrying {retry} times: {e}")
                retry+=1
                if retry>self.retry:
                    # raise e
                    return _prompt,None,None
                time.sleep(self.wait)

        if task_type=="audio2audio":
            wav_bytes = base64.b64decode(response.choices[0].message.audio.data)
            with io.BytesIO(wav_bytes) as buffer:
                waveform, sample_rate = torchaudio.load(buffer)
            return _prompt, pred, (sample_rate, waveform)
        else:
            return _prompt, pred, None


