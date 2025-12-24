import re
import os

import pandas as pd
from loguru import logger
import json
import ast

from ..judge_models import get_judge_model
from .base import AudioBaseDataset

class AudioPretrainDataset(AudioBaseDataset):
    INTERACTIVE = 'Audio-QA'
    TASK = 'Pretrain'
    LANG = None

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, idx: int | str) -> dict:
        if isinstance(idx, str) and idx.isdigit():
            item = self.data[int(idx)]
        if isinstance(idx, int):
            item = self.data[idx]

        if 'audio_path' in item:
            audio_path = item['audio_path']

        if isinstance(audio_path, list):
            for i, p in enumerate(audio_path):
                assert os.path.exists(p), f'Audio file not found: {p}'
        else:
            assert os.path.exists(
                audio_path), f'Audio file not found: {audio_path}'

        # {"index": "sc_standardLogic_000199", 
        # "question": "", 
        # "audio_content": "小红种花时，选了一颗特别漂亮的花苗，精心种在阳台的花盆里，每天按时浇水。几周后，花苗长出了鲜艳的花朵，她非常高兴。", 
        # "audio_path": "data/downloaded_datasets/vcb_bench/pretrain_evaluation/story_continue/audio/context_sc_standardLogic_000199.wav", 
        # "subset": "story_continue", 
        # "task_type": "audio2text", 
        # "answer": "小红每天都欣赏美丽的花朵。", 
        # "answer_audio_path": "data/downloaded_datasets/vcb_bench/pretrain_evaluation/story_continue/audio/trueEnding_sc_standardLogic_000199.wav", 
        # "wrong_answer": "小红发现阳台上的花盆是空的。", 
        # "wrong_answer_audio_path": "data/downloaded_datasets/vcb_bench/pretrain_evaluation/story_continue/audio/falseEnding_sc_standardLogic_000199.wav"}

        question = item['question']

        msg = {'index': item['index'], 'audio': [],
               'text': question, 'meta': self.meta}

        if isinstance(audio_path, list):
            msg['audio'].extend(audio_path)
        elif isinstance(audio_path, str):
            msg['audio'].append(audio_path)

        # 还可以加一个meta信息，默认meta是subset
        if 'subset' in item:
            msg['meta']['subset'] = item['subset']
        if 'meta' in item:
            msg['meta'].update(item['meta'])

        answer = item['answer']
        answer_audio_path = item['answer_audio_path']
        wrong_answer = item['wrong_answer']
        wrong_answer_audio_path = item['wrong_answer_audio_path']
        msg['answer'] = answer
        msg['answer_audio_path'] = answer_audio_path
        msg['wrong_answer'] = wrong_answer
        msg['wrong_answer_audio_path'] = wrong_answer_audio_path

        return msg

    def evaluate(self, eval_file, dump_judge=True, method='default'):

        metrics, judge_results = self.evaluate_inner(eval_file)
        judge_model_name = 'acc'

        model_name = self.get_model_name(eval_file)
        result = self.format_performance(
            model_name, metrics, eval_method=method)

        if dump_judge:
            # dump the judge result to the eval_file
            all_df = []
            for task, judge_result in judge_results.items():
                df = pd.DataFrame(judge_result)
                all_df.append(df)
            all_df = pd.concat(all_df)
            save_file = eval_file.replace(
                '.jsonl', f'_{judge_model_name}_judge.jsonl')
            all_df.to_json(save_file, orient='records',
                           lines=True, force_ascii=False)
        return result

    def evaluate_inner(self, eval_file):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            # import pdb; pdb.set_trace()
            ground_truth = group['answer'].astype(str).to_list()
            preds = group['prediction'].astype(str).to_list()
            # prompts = group['real_prompt'].astype(str).to_list()
            indexes = group['index'].astype(str).to_list()
            results = []
            judge_result = []
            for idx in range(len(preds)):
                pred = preds[idx]
                index = indexes[idx]
                # gt = ground_truth[idx]
                # prompt = prompts[idx]
                pred=ast.literal_eval(pred)
                s2t_pos_loss=pred['s2t_pos_loss']
                s2t_neg_loss=pred['s2t_neg_loss']
                s2s_pos_loss=pred['s2s_pos_loss']
                s2s_neg_loss=pred['s2s_neg_loss']
                res={
                    "type":"_".join(index.split("_")[:2]),
                    "s2t_score": 1 if s2t_pos_loss<s2t_neg_loss else 0,
                    "s2s_score": 1 if s2s_pos_loss<s2s_neg_loss else 0
                    }
                results.append(res)
                org_item = group.iloc[idx].to_dict()
                org_item['judge_result'] = res
                judge_result.append(org_item)
            
            task_result= self.calculate_averages(results)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results
    
    def calculate_averages(self,data_list):
        total_s2t = 0
        total_s2s = 0
        count = 0
        category_sums = {}
        
        for item in data_list:
            s2t_val = item['s2t_score']
            s2s_val = item['s2s_score']
            type_val = item['type']
            
            total_s2t += s2t_val
            total_s2s += s2s_val
            count += 1
            
            if type_val not in category_sums:
                category_sums[type_val] = {'s2t_total': 0, 's2s_total': 0, 'count': 0}
            
            category_sums[type_val]['s2t_total'] += s2t_val
            category_sums[type_val]['s2s_total'] += s2s_val
            category_sums[type_val]['count'] += 1
        
        global_avg = {
            's2t_score': total_s2t / count if count > 0 else 0,
            's2s_score': total_s2s / count if count > 0 else 0
        }
        
        category_avg = {}
        for type_val, sums in category_sums.items():
            category_count = sums['count']
            category_avg[type_val] = {
                's2t_score': sums['s2t_total'] / category_count if category_count > 0 else 0,
                's2s_score': sums['s2s_total'] / category_count if category_count > 0 else 0
            }
        
        return {
            'global_avg': global_avg,
            'category_avg': category_avg
        }

class StoryContinue(AudioPretrainDataset):
    DATASET_NAME = 'story_continue'
    DATASET_SERIES = 'vcb_bench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'
    LANG = "zh"

