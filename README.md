<div align="center">
    <h2>
    VCB-Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents
    </h2>
    <a href="https://arxiv.org/abs/2510.11098"><img src="https://img.shields.io/badge/arXiv-2502.17810-B31B1B.svg" alt="arXiv"></a>
    <a href="https://github.com/Tencent/VCB-Bench"><img src="https://img.shields.io/badge/GitHub-Repo-181717.svg" alt="GitHub"></a>
    <a href="https://huggingface.co/datasets/tencent/VCB-Bench"><img src="https://img.shields.io/badge/Hugging%20Face-Data%20Page-yellow" alt="Hugging Face"></a>

</div>


## Introduction

<b>Voice Chat Bot Bench (VCB Bench)</b> is a high-quality Chinese benchmark built entirely on real human speech. It evaluates large audio language models (LALMs) along three complementary dimensions: 
<br>
(1) <b>Instruction following</b>: Text Instruction Following (TIF), Speech Instruction Following (SIF), English Text Instruction Following (TIF-En), English Speech Instruction Following (SIF-En) and Multi-turn Dialog (MTD);<br>
(2) <b>Knowledge</b>: General Knowledge (GK), Mathematical Logic (ML), Discourse Comprehension (DC) and Story Continuation (SC).<br>
(3) <b>Robustness</b>: Speaker Variations (SV), Environmental Variations (EV), and Content Variations (CV).

## Getting Started

### Installation:

```bash
git clone https://github.com/Tencent/VCB-Bench.git
cd VCB-Bench
pip install -r requirements.txt
```
Note: To evaluate Qwen3-omni, please replace it with the environment it requires.

### Download Dataset:
Download the dataset from [Hugging Face](https://huggingface.co/datasets/tencent/VCB-Bench) and place the 'vcb_bench' into 'data/downloaded_datasets'.

### Evaluation:
This code is adapted from [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit), where you can find more details about the evaluation commands.

(1) Inference + Evaluation:
```
python run_audio.py --model {model_name} --data {data_name}
```
For example:
```
CUDA_VISIBLE_DEVICES=1 python run_audio.py --model Qwen2.5-Omni-7B --data general_knowledge
```

(2) Only Inference:
```
python run_audio.py --model {model_name} --data {data_name} --skip-eval
```
For example:
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_audio.py --model  StepAudio  --data continuation_en  creation_en  empathy_en  recommendation_en  rewriting_en  safety_en  simulation_en emotional_control_en  language_control_en  non_verbal_vocalization_en  pacing_control_en  style_control_en  volume_control_en --skip-eval 
```
(3) Only Evaluation:
```
python run_audio.py --model {model_name} --data {data_name} --reeval
```
For example:
```
CUDA_VISIBLE_DEVICES=2 nohup python run_audio.py --model  Mimo-Audio --data continuation  creation  empathy --reeval
```
(4) Inference + ASR + Evaluation:
```
python run_audio.py --model {model_name} --data {data_name} --wasr
```
For example:
```
CUDA_VISIBLE_DEVICES=3 python run_audio.py --model  StepAudio2 --data rewriting  safety  simulation  continuation_en  --wasr 
```

### Format Result:
```
python sumup_eval.py --model {model_name}
```
```
python sumup_eval.py --model {model_name} --export_excel --output_file my_results.xlsx
```

## Supported Datasets and Models
(1) Locate the dataset you need to evaluate from the Data Name column in the Datasets table, and populate the {data_name} parameter in the evaluation command accordingly.<br>
(2) Each dataset in the SV, EV, and CV sections has a corresponding comparison dataset named "{data_name}_cmp", following the specified naming convention.<br>
(3) Identify the model you intend to evaluate from the Model Name column in the Models table, and insert the appropriate {model_name} into the evaluation command.
### Datasets:
<table>
  <thead>
    <tr>
      <th>Data Type</th>
      <th>Data Name</th>
      <th>Detail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="category" rowspan="7">TIF</td>
      <td>continuation</td>
      <td>-</td>
    </tr>
    <tr>
      <td>creation</td>
      <td>-</td>
    </tr>
    <tr>
      <td>empathy</td>
      <td>-</td>
    </tr>
    <tr>
      <td>recommendation</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rewriting</td>
      <td>-</td>
    </tr>
    <tr>
      <td>safety</td>
      <td>-</td>
    </tr>
    <tr>
      <td>simulation</td>
      <td>-</td>
    </tr>
    <tr>
      <td class="category" rowspan="7">TIF-En</td>
      <td>continuation_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>creation_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>empathy_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>recommendation_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rewriting_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>safety_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>simulation_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td class="category" rowspan="6">SIF</td>
      <td>emotional_control</td>
      <td>-</td>
    </tr>
    <tr>
      <td>language_control</td>
      <td>-</td>
    </tr>
    <tr>
      <td>non_verbal_vocalization</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pacing_control</td>
      <td>-</td>
    </tr>
    <tr>
      <td>style_control</td>
      <td>-</td>
    </tr>
    <tr>
      <td>volume_control</td>
      <td>-</td>
    </tr>
    <tr>
      <td class="category" rowspan="6">SIF-En</td>
      <td>emotional_control_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>language_control_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>non_verbal_vocalization_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pacing_control_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>style_control_en</td>
      <td>-</td>
    </tr>
    <tr>
      <td>volume_control_en</td>
      <td>-</td>
    </tr> 
    <tr>
      <td class="category" rowspan="3">MTD</td>
      <td>progression</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backtracking</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transition</td>
      <td>-</td>
    </tr> 
    <tr>
      <td class="category" rowspan="1">GK</td>
      <td>general_knowledge</td>
      <td>mathematics, geography, politics, chemistry, biology, law, physics, history, medicine, economics, sports, culture</td>
    </tr>
    <tr>
      <td class="category" rowspan="3">ML</td>
      <td>basic_math</td>
      <td>-</td>
    </tr>
    <tr>
      <td>math</td>
      <td>-</td>
    </tr>
    <tr>
      <td>logical_reasoning</td>
      <td>analysis, induction, analogy, logic</td>
    </tr>
    <tr>
      <td class="category" rowspan="1">DC</td>
      <td>discourse_comprehension</td>
      <td>inference, induction, analysis</td>
    </tr>
    <tr>
      <td class="category" rowspan="4">SV</td>
      <td>age</td>
      <td>child, elder</td>
    </tr>
    <tr>
      <td>accent</td>
      <td>tianjin, beijing, dongbei, sichuan</td>
    </tr>
    <tr>
      <td>volume</td>
      <td>down, up</td>
    </tr>
    <tr>
      <td>speed</td>
      <td>-</td>
    </tr>
    <tr>
      <td class="category" rowspan="3">EV</td>
      <td>non_vocal_noise</td>
      <td>echo, outdoors, far_field</td>
    </tr>
    <tr>
      <td>vocal_noise</td>
      <td>TV_playback, background_chat, vocal_music, voice_announcement</td>
    </tr>
    <tr>
      <td>unstable_signal</td>
      <td>-</td>
    </tr>
    <tr>
      <td class="category" rowspan="5">CV</td>
      <td>casual_talk</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mispronunciation</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grammatical_error</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topic_shift</td>
      <td>-</td>
    </tr>
    <tr>
      <td>code_switching</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

### Models:

<table>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>Model Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-type" rowspan="10">Chat Model</td>
      <td>Qwen2-Audio-7B-Instruct</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni-7B</td>
    </tr>
    <tr>
      <td>Baichuan-Audio-Chat</td>
    </tr>
    <tr>
      <td>GLM4-Voice</td>
    </tr>
    <tr>
      <td>Kimi-Audio</td>
    </tr>
    <tr>
      <td>Mimo-Audio</td>
    </tr>
    <tr>
      <td>StepAudio</td>
    </tr>
    <tr>
      <td>StepAudio2</td>
    </tr>
    <tr>
      <td>GPT4O-Audio</td>
    </tr>
    <tr>
      <td>Qwen3-Omni-Instruct</td>
    </tr>
    <tr>
      <td class="model-type" rowspan="4">Pretrain Model</td>
      <td>Qwen2-Audio-7B</td>
    </tr>
    <tr>
      <td>Baichuan-Audio</td>
    </tr>
    <tr>
      <td>Kimi-Audio-Base</td>
    </tr>
    <tr>
      <td>StepAudio2-Base</td>
    </tr>
  </tbody>
</table>

## Acknowledge
We borrow some code from [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit), [GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice), [Baichuan-Audio](https://github.com/baichuan-inc/Baichuan-Audio), [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio), [Mimo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio), [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2), and [StepAudio](https://github.com/stepfun-ai/Step-Audio).

## Citation
```
@misc{hu2025vcbbenchevaluationbenchmark,
      title={VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents}, 
      author={Jiliang Hu and Wenfu Wang and Zuchao Li and Chenxing Li and Yiyang Zhao and Hanzhao Li and Liqiang Zhang and Meng Yu and Dong Yu},
      year={2025},
      eprint={2510.11098},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.11098}, 
}
```

