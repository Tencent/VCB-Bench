import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import jsonlines
from funasr import AutoModel
from tqdm import tqdm

def set_whisper(model_dir="openai"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        os.path.join(model_dir,"whisper-large-v3"), torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(os.path.join(model_dir,"whisper-large-v3"))

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "en", "return_timestamps": True},
        return_timestamps=True,
    )
    return pipe

def set_paraformer(model_dir="funasr"):
    pipe = AutoModel(
        model=os.path.join(model_dir,"paraformer-zh"),
        model_revision="v2.0.4",
        vad_model=os.path.join(model_dir,"fsmn-vad"),
        vad_model_revision="v2.0.4",
        punc_model=os.path.join(model_dir,"ct-punc"),
        punc_model_revision="v2.0.4",
        hub="ms",
        # spk_model="cam++", spk_model_revision="v2.0.2",
    )
    return pipe

def do_s2t(dest_file,lang):
    if lang not in ["zh", "en"]:
        raise NotImplementedError("lang support only 'zh' and 'en' for now.")
    if lang == "en":
        pipe = set_whisper()
    elif lang == "zh":
        pipe = set_paraformer()
    # import pdb;pdb.set_trace()
    src_file=dest_file.replace("_wasr.jsonl", ".jsonl")
    audio_root=os.path.join(os.path.dirname(dest_file), "audio")
    wf=jsonlines.open(dest_file, mode="w")
    with open(src_file, mode="r", encoding='utf-8') as f:
        # 首先计算总行数以设置进度条
        total_lines = sum(1 for _ in jsonlines.Reader(f))
        # 重新打开文件进行实际处理
        f.seek(0)
        for item in tqdm(jsonlines.Reader(f), total=total_lines, desc="Processing ASR"):
            audio_path=os.path.join(audio_root, str(item["index"])+".wav")
            if os.path.exists(audio_path):
                if lang == "en":
                    result = pipe([audio_path], batch_size=1)
                elif lang == "zh":
                    result = pipe.generate(input=audio_path)
                try:
                    item["prediction"] = result[0]["text"].strip()
                except:
                    item["prediction"] = ""
                wf.write(item)
    wf.close()
