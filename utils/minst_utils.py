import torch
import pytorch_lightning.metrics as metrics


def get_mnist_pair(pred, true) -> (list, list):
    """
    获取两个预测标签和真实标签对应的list

    :param pred: 预测值 torch.Size([B,10])
    :param true: 真是值 torch.Size([B])
    :return: [pre_list, true_list]
    """
    y = torch.argmax(pred, dim=-1)
    y_list = y.detach().cpu().numpy().tolist()
    true_list = true.detach().cpu().numpy().tolist()
    return y_list, true_list


if __name__ == '__main__':
    metrics.Accuracy()