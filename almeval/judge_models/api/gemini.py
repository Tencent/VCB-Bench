import json
import os
import sys
import traceback
import requests
import aiohttp
import httpx
import numpy as np
from loguru import logger
import base64
import asyncio
from google.oauth2 import service_account
from google.auth.transport.requests import Request as AuthRequest
from .base import BaseAPI

def get_access_token(credentials_path):
    """
    Get OAuth2 Access Token
    """
    scopes = ['https://www.googleapis.com/auth/cloud-platform']
    creds = service_account.Credentials.from_service_account_file(
    credentials_path, scopes=scopes
    )
    creds.refresh(AuthRequest())
    return creds.token


def audio2base64(audio_path):
  wav_data = open(audio_path, "rb").read()
  encoded_string = base64.b64encode(wav_data).decode('utf-8')
  return encoded_string


class GeminiWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gemini-2.5-pro',
                 retry: int = 50,
                 wait: int = 10,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0.4,
                 top_p: float = 0.4,
                 top_k: int = 32,
                 timeout: int = 600,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 use_azure: bool = True,
                 ocr_provider: str = None,
                 **kwargs):

        self.model = model
        self.api_base = api_base
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_azure=use_azure
        self.ocr_provider = ocr_provider

        self.key = key
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry,
                         system_prompt=system_prompt, verbose=verbose, **kwargs)
        
        credentials_path = os.environ['GOOGLE_CREDENTIALS_PATH']
        project_id = os.environ['GOOGLE_PROJECT_ID']
        location = kwargs.pop('location', 'us-central1')
        self.access_token = get_access_token(credentials_path)
        self.api_base = self.api_base.format(PROJECT_ID=project_id, LOCATION=location, MODEL_ID=model)

    async def generate_inner(self, inputs, **kwargs) -> str:
        audio_path = kwargs.pop('audio_path', None)
        #print(audio_path)
        if not audio_path:
            raise Exception("Error: audio to be evaluated is not found")

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": inputs[0]["value"]
                        },
                        {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": audio2base64(audio_path)
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                # "thinking_config": {
                #      "thinking_level": "LOW" # 3 pro
                #  },
                "maxOutputTokens": kwargs.pop('max_tokens', self.max_tokens),
                "temperature": kwargs.pop('temperature', self.temperature),
                "topP": kwargs.pop('top_p', self.top_p),
                "topK": kwargs.pop('top_k', self.top_k)
            },
        }
        #async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        #breakpoint()
        async with httpx.AsyncClient(timeout=self.timeout * 1.1,verify=False) as client:
            response = await client.post(
                    self.api_base,
                    headers=headers, json=payload)
            #ret_code = response.status
            ret_code = response.status_code
            #print(ret_code)
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            answer = self.fail_msg
            # response_text = await response.text()
        try:
            # import pdb;pdb.set_trace()
            response_text = response.text
            resp_json = json.loads(response_text)
            candidates = resp_json.get('candidates', [])
            content_parts = candidates[0].get('content', {}).get('parts', [])
            answer = ""
            for part in content_parts:
                if 'text' in part:
                    answer += part['text']
        except Exception as e:
            print(f'Error: {e}')
            print(f'Traceback:\n{traceback.format_exc()}')
            print(f'Response:\n{response.text}')
            answer = self.fail_msg
            # import pdb; pdb.set_trace()
        return ret_code, answer, response
