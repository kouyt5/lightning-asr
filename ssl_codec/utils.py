from typing import List, Tuple, Union
import numpy as np


import torch


def sum_logprob(evaluated_tensors: torch.Tensor(), encode_len:List):
    """
    计算语音识别的置信度
    :params
        evaluated_tensors: 语音识别的对数概率 B*T*E
    """
    log_probs, index = torch.max(evaluated_tensors, dim=-1)
    results_list = []

    for i in range(index.shape[0]):
        sum = -1e-5
        count = 0.
        vocab_size = evaluated_tensors.size(2)
        for j in range(encode_len[i]):
            if index[i][j].item() == vocab_size:
                continue
            sum += log_probs[i][j].item()
            count += 1
        avg_prob = sum / (count + 1e-6)
        results_list.append(-avg_prob)
    return results_list

def seq_sum_logprob(data:Tuple[int, torch.Tensor, int]):
    """
    用于多线程计算对数概率
    :params
        data: (1, Tensor, 500)
    """
    log_probs, index = torch.max(data[1], dim=-1)  # input 343*4334
    sum = -1e-5
    count = 0.
    vocab_size = data[1].size(1)
    for j in range(data[2]):
        if index[j].item() == vocab_size:
            continue
        sum += log_probs[j].item()
        count += 1
    avg_prob = sum / (count + 1e-6)
    return data[0], -avg_prob

def seq_sum_logprob_np(data:Tuple[int, torch.Tensor, int]):
    """
    用于多线程计算对数概率 数据为numpy
    :params
        data: (1, Tensor, 500)
    """
    # log_probs, index = torch.max(data[1], dim=-1)  # input 343*4334
    indicts = np.argmax(data[1], axis=-1)
    log_probs = [data[1][i][indicts[i]] for i in range(data[1].shape[0])]
    sum = -1e-5
    count = 0.
    vocab_size = data[1].shape[1]
    for j in range(data[2]):
        if indicts[j] == vocab_size:
            continue
        sum += log_probs[j]
        count += 1
    avg_prob = sum / (count + 1e-6)
    return data[0], -avg_prob

if __name__ == '__main__':
    moke_data = torch.randn((412, 4334))
    seq_sum_logprob_np((1,moke_data.numpy(), 123))