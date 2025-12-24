from abc import abstractmethod
import math
import librosa
import torch
from torch.nn import functional as F
from loguru import logger


class BaseModel:

    NAME = None

    @abstractmethod
    def generate_inner(self, msg: dict) -> (str, str):
        raise NotImplementedError

    @staticmethod
    def check_audio_legal(audio_path: str | list[str], max_duration: float = 60) -> bool:
        """by default, we discard audio longer than 60s. subclasses can override this method (depends on model requirements)
        """
        if isinstance(audio_path, str):
            duration = librosa.get_duration(path=audio_path)
            if duration > max_duration or duration < 0.1:
                return False
        else:
            for path in audio_path:
                duration = librosa.get_duration(path=path)
                if duration > max_duration or duration < 0.1:
                    return False
        return True

    @torch.inference_mode()
    def __call__(self, msg: dict) -> str:
        if not self.check_audio_legal(msg['audio']):
            logger.warning(
                f'dataset: {msg["meta"]["dataset_name"]}, audio: {msg["audio"]}, duration exceeds 60s limit, skipping this sample')
            return msg['text'], None
        return self.generate_inner(msg)
    
    def mask_loss_ppl(self, logits, input_ids):
        # 使用input_ids作为labels（无监督训练）
        labels = input_ids
        
        # 获取特殊token掩码
        # _, sp_mask, _ = model.model.get_multimodal_mask(
        #     input_ids, 
        #     model.config.audio_config.audio_pad_token_id, 
        #     model.config.multimodal_special_token_list
        # )
        
        # 移位操作：预测下一个token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 创建有效掩码
        valid_mask = torch.gt(shift_labels, -1) 
        
        # 调整特殊token掩码以匹配移位后的序列
        # sp_mask = sp_mask[..., 1:].contiguous()
        # 若需要屏蔽特殊token
        # valid_mask = valid_mask & sp_mask
        
        # 所有token都计算损失
        
        # 扁平化logits和labels
        # import pdb;pdb.set_trace()
        shift_logits_flat = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels_flat = shift_labels.view(-1)
        shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)
        
        # 计算损失（忽略索引-100，但由于无pad，不会忽略任何位置）
        flatten_loss = F.cross_entropy(
            shift_logits_flat, 
            shift_labels_flat, 
            ignore_index=-100, 
            reduction='none'
        )
        
        # 计算总损失和有效token数
        loss_sum = flatten_loss.sum()
        valid_count = valid_mask.sum() + 1e-10  # 防止除零
        
        # 计算平均损失和困惑度
        loss_mean = (loss_sum / valid_count).item()
        ppl = math.exp(loss_mean)
        
        return loss_mean, ppl
