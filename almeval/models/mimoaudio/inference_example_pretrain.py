# Copyright 2025 Xiaomi Corporation.
from src.mimo_audio.mimo_audio import MimoAudio

model_path = "models/MiMo-Audio-7B-Base"
tokenizer_path = "models/MiMo-Audio-Tokenizer"


model = MimoAudio(model_path, tokenizer_path)


# in context learning: speech-to-speech generation
instruction = "Convert the timbre of the input speech to target timbre."

input_audio = "examples/ESD/0013_000200.wav"
prompt_examples = [
    {
        "input_audio": "examples/ESD/0013_000139.wav",
        "output_audio": "examples/ESD/0019_000139.wav",
        "output_transcription": "Cuckoos is downheaded and crying.",
    },
    {
        "input_audio": "examples/ESD/0013_000963.wav",
        "output_audio": "examples/ESD/0019_000963.wav",
        "output_transcription": "She said in subdued voice.",
    },
    {
        "input_audio": "examples/ESD/0013_000559.wav",
        "output_audio": "examples/ESD/0019_000559.wav",
        "output_transcription": "A raging fire was-in his eyes.",
    },
    {
        "input_audio": "examples/ESD/0013_001142.wav",
        "output_audio": "examples/ESD/0019_001142.wav",
        "output_transcription": "Does the one that wins get the crowned?",
    },
    {
        "input_audio": "examples/ESD/0013_000769.wav",
        "output_audio": "examples/ESD/0019_000769.wav",
        "output_transcription": "Not much use is it, sam?",
    },
]

output_audio_path = "examples/in_context_learning_s2s.wav"
text_channel_output = model.in_context_learning_s2s(instruction, prompt_examples, input_audio, max_new_tokens=8192, output_audio_path=output_audio_path)