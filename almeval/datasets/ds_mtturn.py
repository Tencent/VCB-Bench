import re
import os
import ast
import pandas as pd
from collections import defaultdict

from ..judge_models import get_judge_model
from .base import AudioBaseDataset

Multi_Turn_Prompt = """
I need your help to evaluate the performance of several models in the multi-round speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s latest response [Response] based on the provided dialogue history [History], which includes transcriptions of both the user’s inputs and the model’s previous responses.

Please evaluate the response on a scale of 1 to 5:
1 point: Response is irrelevant or nonsensical. Or response ignores previous turns, leading to confusion or irrelevance.
2 points: Some answers are relevant but many lack detail or completeness. Frequently loses track of the conversation, with response that is not aligned with earlier turns.
3 points: Response is mostly relevant and coherent, though occasional lapses in depth. The model follows the conversation, but may occasionally forget important details from earlier turns.
4 points: Response is clear, relevant, and detailed. Generally keeps track of the conversation, with minor lapses.
5 points: Response is clear, relevant, and detailed. Flawlessly integrates context across all rounds, ensuring natural conversation flow, creating an engaging experience.

Below are the transcription of dialogue history and model’s response:
### [History]: 
{question}
### [Response]: 
{prediction}

Please output only one score for the whole conversation without anything else.
You don’t need to provide any explanations.
"""


class AudioMtTurnDataset(AudioBaseDataset):
    INTERACTIVE = 'Audio-QA'
    TASK = 'Multi-Turn'

    def build_prompt(self, idx: int | str) -> dict:
        """Construct a user conversation from a data source. Currently, we only consider single-turn conversations.

        The user's text may be single, but may require multiple audio inputs, e.g., "Which audio is louder?"
            {
                'audio': [audio_path],
                'text': question,
            }
        The question can also be empty.
        Sometimes, the model needs additional meta information to decide how to construct the prompt, so we return the model's meta information as well.
        - meta adds subset information to indicate which subset this msg belongs to
        - if there is a meta field in msg, it will be merged into msg['meta']
        """
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

        question = item['question']

        msg = {'index': item['index'], 'audio': [],
               'text': question, 'task_type':item["task_type"], 'meta': self.meta}

        if isinstance(audio_path, list):
            msg['audio'].extend(audio_path)
        elif isinstance(audio_path, str):
            msg['audio'].append(audio_path)

        # 还可以加一个meta信息，默认meta是subset
        if 'subset' in item:
            msg['meta']['subset'] = item['subset']
        if 'meta' in item:
            msg['meta'].update(item['meta'])
        msg['meta']['context'] = item["audio_content"]
        return msg

    def evaluate(self, eval_file, dump_judge=True, method='default'):
        if method == 'default':
            method = 'gpt-4o-mini'
        else:
            raise NotImplementedError
        
        judge_model = get_judge_model(method)
        metrics, judge_results = self.evaluate_llm(
            eval_file, judge_model)
        judge_model_name = judge_model.model

        model_name = self.get_model_name(eval_file)
        result = self.format_performance(
            model_name, metrics, eval_method=method)

        if dump_judge and judge_results is not None:
            # dump the judge result to the eval_file
            all_df = []
            for task, judge_result in judge_results.items():
                if judge_result is None:
                    continue
                df = pd.DataFrame(judge_result)
                all_df.append(df)
            all_df = pd.concat(all_df)
            save_file = eval_file.replace(
                '.jsonl', f'_{judge_model_name}_judge.jsonl')
            df.to_json(save_file, orient='records', lines=True)
        return result

    @staticmethod
    def extract_rating(llm_output):
        """
        Extracts the rating in the format [[number]] from the LLM output.

        Args:
        - llm_output (str): The response from the LLM containing the evaluation and rating.

        Returns:
        - int: The extracted rating, or None if the rating is not found.
        """
        # Define the regular expression pattern to match the rating in the format [[number]]
        pattern = r'\[\[(\d+)\]\]'

        # Search for the pattern in the LLM output
        match = re.search(pattern, llm_output)

        if match:
            # Convert the matched rating to an integer and return it
            return int(match.group(1))
        else:
            # Return None if no rating is found
            # return None
            raise NotImplementedError

    def evaluate_llm(self, eval_file, judge_model=None, n_times=1):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        # import pdb;pdb.set_trace()
        for task, group in df.groupby('subset'):
            question = self.get_LLM_query(group)
            question = self.merge_dialog_history(question)
            pred = group['prediction'].astype(str).to_list()
            # duplicate pred and question times
            pred = pred * n_times
            question = question * n_times
            results = self.run_llm_judge(
                judge_model, Multi_Turn_Prompt, pred=pred, question=question, temperature=0.5, top_p=0.95)
            results = results[:len(results) // n_times]
            pred = pred[:len(pred) // n_times]
            question = question[:len(question) // n_times]
            # score = 0.
            # cnt = 0
            # invalid = 0
            judge_result = []

            # {"index": "huisu0238-1", 
            #  "question": "", 
            #  "audio_content": ["小孩晚上老是睡得晚，有什么办法让他早点睡？", 
            #                    "建议建立固定睡前程序，比如九点洗澡，九点半读书，十点上床睡觉。", 
            #                    "固定睡前程序怎么设定具体时间？"], 
            #  "audio_path": ["data/downloaded_datasets/vcb_bench/multi_turn_dialog/backtracking/audio/huisu0238_984.wav", 
            #                 "data/downloaded_datasets/vcb_bench/multi_turn_dialog/backtracking/audio/huisu0238_985.wav", 
            #                 "data/downloaded_datasets/vcb_bench/multi_turn_dialog/backtracking/audio/huisu0238_986.wav"], 
            # "subset": "backtracking", "task_type": "audio2text", "mt_meta": {"rounds": "3", "this_round": "2", "score_ratio": "1:1:2"}
            # }

            scores = []
            # import pdb;pdb.set_trace()
            for i, res in results:
                org_item = group.iloc[i].to_dict()
                org_item['judge_result'] = res
                judge_result.append(org_item)
                conversation_id=org_item["index"].split("-")[0]
                round_id=int(org_item["mt_meta"]["this_round"])
                total_ratio=sum([int(i) for i in org_item["mt_meta"]["score_ratio"].split(":")])
                score_ratio=int(org_item["mt_meta"]["score_ratio"].split(":")[round_id-1])/total_ratio
                try:
                    # if output lot of text, try to get the first score 
                    try:
                        res = res.strip()
                        new_score = float(res)
                    except Exception:
                        print(f'Fail to convert {res} to score. skip.')
                        new_score = self.extract_rating(res)
                    scores.append({
                        "id": conversation_id,
                        "score_ratio": score_ratio,
                        "score": new_score
                        })
                    # score += new_score
                    # cnt += 1
                except Exception:
                    # invalid += 1
                    scores.append({
                        "id": conversation_id,
                        "score_ratio": score_ratio,
                        "score": None,
                    })
                    print(f'Fail to convert {res} to score. skip.')

            score,valid_cv_num,invalid_cv_num=self.calculate_average_score(scores)
            task_result = {
                'score': score,
                'total': valid_cv_num,
                'invalid': invalid_cv_num
            }
            metrics[task] = task_result
            judge_results[task] = judge_result
            print(f'{task} result: {task_result}')

        return metrics, judge_results
    
    def calculate_average_score(self, data_list):
        # from collections import defaultdict
        
        # 按id分组
        groups = defaultdict(list)
        for item in data_list:
            conv_id = item['id']
            groups[conv_id].append(item)
        
        # 计算每个对话的总分数，同时检查是否有None值
        valid_conversation_totals = []
        invalid_conversation_count = 0
        
        for conv_id, items in groups.items():
            # 检查对话中是否有任何元素的分数为None
            has_none = any(item['score'] is None for item in items)
            
            if has_none:
                invalid_conversation_count += 1
                continue  # 跳过这个对话
            
            # 计算有效对话的总分数
            total_score = 0
            for item in items:
                total_score += item['score'] * item['score_ratio']
            valid_conversation_totals.append(total_score)
        
        # 计算所有有效对话的平均分数
        if valid_conversation_totals:
            average_score = round((sum(valid_conversation_totals)/len(valid_conversation_totals))*20,2)
        else:
            average_score = 0
        
        valid_conversation_count = len(valid_conversation_totals)
        
        return average_score, valid_conversation_count, invalid_conversation_count
    
    def merge_dialog_history(self, questions):
        new_questions=[]
        for question in questions:
            new_question=""
            question=ast.literal_eval(question)
            for i,q in enumerate(question):
                if(i%2==0):
                    new_question+=f"User: {q}\n"
                else:
                    new_question+=f"Model: {q}\n"
            new_questions.append(new_question)
        return new_questions



class ProgressionDataset(AudioMtTurnDataset):
    DATASET_NAME = 'progression'
    DATASET_SERIES = 'vcb_bench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'
    LANG = "zh"

class BacktrackingDataset(AudioMtTurnDataset):
    DATASET_NAME = 'backtracking'
    DATASET_SERIES = 'vcb_bench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'
    LANG = "zh"

class TaskTransitionDataset(AudioMtTurnDataset):
    DATASET_NAME = 'transition'
    DATASET_SERIES = 'vcb_bench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'
    LANG = "zh"
