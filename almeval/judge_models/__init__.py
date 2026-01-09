import os

from .api import OpenAIWrapper
from .api import OpenAIAudioWrapper
from .api import GeminiWrapper

async def judge_response(prompt: str, judge_model: OpenAIWrapper,
                         temperature=0.0,
                         top_p=0.95,
                         max_tokens=1024,
                         frequency_penalty=0,
                         presence_penalty=0,
                         stop=None,
                         audio_path=None):
    judge_result_str = ''
    try:
        judge_result_str = await judge_model.generate(prompt,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      frequency_penalty=frequency_penalty,
                                                      presence_penalty=presence_penalty, stop=stop,audio_path=audio_path)
        judge_result_str = judge_result_str.strip()
        return judge_result_str

    except Exception as e:
        print(f'Judge failed!: {e}')
        print(f'Output:\n{judge_result_str}')
        return None


def get_gpt4o_model():
    cur_judge_kwargs = {}
    cur_judge_kwargs['model'] = 'gpt-4o'
    cur_judge_kwargs['api_base'] = 'https://api.openai.com/v1/chat/completions'
    assert os.environ['OPENAI_API_KEY']

    judge_model = OpenAIWrapper(**cur_judge_kwargs)
    return judge_model


def get_gpt4o_mini_model():
    cur_judge_kwargs = {}
    cur_judge_kwargs['model'] = 'gpt-4o-mini'
    cur_judge_kwargs['api_base'] = 'https://api.openai.com/v1/chat/completions'
    cur_judge_kwargs['system_prompt'] = "You are a helpful assistant who tries to help answer the user's question."
    assert os.environ['OPENAI_API_KEY']
    judge_model = OpenAIWrapper(**cur_judge_kwargs)
    return judge_model


def get_gpt4o_audio_model():
    cur_judge_kwargs = {}
    cur_judge_kwargs['model'] = 'gpt-4o-audio-preview'
    cur_judge_kwargs['api_base'] = 'https://api.openai.com/v1/chat/completions'
    cur_judge_kwargs['system_prompt'] = "You are a helpful assistant."
    assert os.environ['OPENAI_API_KEY']
    judge_model = OpenAIAudioWrapper(**cur_judge_kwargs)
    return judge_model

def get_gemini_2_5_pro_model():
    cur_judge_kwargs = {}
    cur_judge_kwargs['model'] = 'gemini-2.5-pro'
    cur_judge_kwargs['location'] = 'us-central1'
    cur_judge_kwargs['api_base'] = "https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    assert os.environ['GOOGLE_CREDENTIALS_PATH'], "GOOGLE_CREDENTIALS_PATH is not set"
    assert os.environ['GOOGLE_PROJECT_ID'], "GOOGLE_PROJECT_ID is not set"

    judge_model = GeminiWrapper(**cur_judge_kwargs)
    return judge_model


def get_judge_model(method='default'):
    if method == 'default':
        method = 'gpt-4o-mini'
    elif method == 'gpt-4o':
        return get_gpt4o_model()
    elif method == 'gpt-4o-mini':
        return get_gpt4o_mini_model()
    elif method == 'gpt-4o-audio':
        return get_gpt4o_audio_model()
    elif method == 'gemini-2.5-pro':
        return get_gemini_2_5_pro_model()
    else:
        raise ValueError(f'Unsupported evaluation method: {method}')

