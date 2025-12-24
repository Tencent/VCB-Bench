# Copyright 2025 Xiaomi Corporation.
from src.mimo_audio.mimo_audio import MimoAudio

model_path = "models/MiMo-Audio-7B-Instruct"
tokenizer_path = "models/MiMo-Audio-Tokenizer"


model = MimoAudio(model_path, tokenizer_path)


# tts
text = "今天天气真好"
output_audio_path = "examples/tts.wav"
text_channel_output = model.tts_sft(text, output_audio_path)


# instruct tts
text = "今天天气真好"
output_audio_path = "examples/instruct_tts.wav"
instruct = "用小孩子的声音开心的说"
text_channel_output = model.tts_sft(text, output_audio_path, instruct=instruct)


# natural instruction tts
text = "用气喘吁吁的年轻男性声音说：我跑不动了，你等等我！"
output_audio_path = "examples/natural_instruction_tts.wav"
text_channel_output = model.tts_sft(text, output_audio_path, read_text_only=False)


# audio understanding
audio_path = "examples/spoken_dialogue_assistant_turn_1.wav"
text = "Summarize the audio."
text_channel_output = model.audio_understanding_sft(audio_path, text)


# audio understanding with thinking
audio_path = "examples/spoken_dialogue_assistant_turn_1.wav"
text = "Summarize the audio."
text_channel_output = model.audio_understanding_sft(audio_path, text, thinking=True)


# spoken dialogue
first_turn_text_response = "我没办法获取实时的天气信息。不过呢，你可以试试几个方法来查看今天的天气。首先，你可以用手机自带的天气功能，比如苹果手机的天气应用，或者直接在系统设置里查看。其次，你也可以用一些专业的天气服务，像是国外的AccuWeather、Weather.com，或者国内的中国天气网、墨迹天气等等。再有就是，你还可以在谷歌或者百度里直接搜索你所在的城市加上天气这两个字。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。"
message_list = [
    {"role": "user", "content": "examples/今天天气如何.mp3"},
    {"role": "assistant", "content": {"text": first_turn_text_response, "audio": "examples/spoken_dialogue_assistant_turn_1.wav"}},
    {"role": "user", "content": "examples/北京.mp3"},
]
output_audio_path = "examples/spoken_dialogue_assistant_turn_2.wav"
text_channel_output = model.spoken_dialogue_sft_multiturn(message_list, output_audio_path=output_audio_path, system_prompt=None, prompt_speech="examples/prompt_speech_zh_m.wav")
text_channel_output = text_channel_output.split("<|eot|>")[0].replace(".....", "")
print(text_channel_output)


# speech-to-text dialogue
message_list = [
    {"role": "user", "content": "./examples/今天天气如何.mp3"},
    {"role": "assistant", "content": "你好，我没办法获取实时的天气信息。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。"},
    {"role": "user", "content": "./examples/北京.mp3"},
]
text_channel_output = model.speech2text_dialogue_sft_multiturn(message_list, thinking=True)


# text dialogue

message_list = [
    {"role": "user", "content": "可以给我介绍一些中国的旅游景点吗？"},
    {"role": "assistant", "content": "你好，您想去哪个城市旅游呢？"},
    {"role": "user", "content": "北京"},
]
text_channel_output = model.text_dialogue_sft_multiturn(message_list, thinking=True)