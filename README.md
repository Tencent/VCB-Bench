<h1 align="center">
    VCB Bench Evalkit
</h1>


## Introduction

Voice Chat Bot Bench (VCB Bench) is a high-quality Chinese benchmark built entirely on real human speech. It evaluates large audio language models (LALMs) along three complementary dimensions: 
<br>
(1) Instruction following: Text Instruction Following (TIF), Speech Instruction Following (SIF), English Text Instruction Following (TIF-En), English Speech Instruction Following (SIF-En) and Multi-turn Dialog (MTD);<br>
(2) Knowledge: General Knowledge (GK), Mathematical Logic (ML), Discourse Comprehension (DC) and Story Continuation (SC).<br>
(3) Robustness: Speaker Variations (SV), Environmental Variations (EV), and Content Variations (CV).

## Getting Started

### Installation:

```bash
git clone https://github.com/193746/VCB-Bench-Evalkit.git
cd VCB-Bench-Evalkit
pip install -r requirements.txt
```
Note: To evaluate Qwen3-omni, please replace it with the environment it requires.

### Download Dataset:

```bash
cd data/downloaded_datasets
tar zxvf vcbbench.tar.gz
```

### Evaluation:
This code is adapted from [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio-Evalkit), where you can find more details about the evaluation commands.

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


