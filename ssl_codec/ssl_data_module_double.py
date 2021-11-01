import logging
from typing import Optional, List, Union
import sys, os
sys.path.append(os.path.abspath('.'))
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import os
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import random
import numpy as np
import pickle

from ssl_codec.convert_manifestwav2pkl import Model, Wav2Vec2Extractor


"""
无监督训练数据集
"""
class SSLAudioDataset(Dataset):
    def __init__(self, manifest_path: list, labels, max_duration=16, mask=False, win_len=0.025, sr=16000, 
                    ssl_folder:str=None, on_the_flying:bool=True):
        """
        Args:
            manifest_path: 原始音频文件路径标签对应表
            labels: 标签
            max_duration: 最大时长
            mask: 是否对特征做mask
            win_len: 窗长
            sr=16000
            ssl_folder: 存放无监督模型提取到的音频特征的文件夹，名字和音频名相同，如果未指定，使用on the flying方式在线提取特征
            ssl_model_retrain: 自监督模型是否需要重新训练
        """

        self.on_the_flying = on_the_flying
        
        self.datasets = []
        self.labels = labels
        self.mask = mask
        self.ssl_folder = ssl_folder
        for item in manifest_path:
            with open(item, encoding='utf-8') as f:
                for line in f.readlines():
                    data = json.loads(line, encoding='utf-8')
                    if data['duration'] > max_duration:
                        continue
                    if ssl_folder is not None and not self.on_the_flying:  # 如果没有指定ssl提取的特征的文件夹
                        # 获取提取到的特征的路径
                        data['ssl_audio_filepath'] = os.path.join(ssl_folder, data['audio_filepath'].split('/')[-1].split('.wav')[0]+'.pkl')
                    self.datasets.append(data)
        self.index2char = dict([(i, labels[i]) for i in range(len(labels))])
        self.char2index = dict([(labels[i], i) for i in range(len(labels))])
        self.pesudo_datasets = []  # 伪标签数据集，默认为空
        self.audio_parser = AudioParser(win_len=win_len, sr=sr, hop_len=0.02)

        # self.audio_parser = AudioParser(win_len=win_len, sr=sr)

    def __getitem__(self, index):
        data = self.datasets[index]
        text2id = [self.char2index[char] for char in data['text']]
        if self.ssl_folder is not None and self.on_the_flying is False:
            return data["ssl_audio_filepath"], text2id, self.audio_parser.parse_audio(data['audio_filepath'], mask=self.mask), self.mask
        return data["audio_filepath"], text2id, self.audio_parser.parse_audio(data['audio_filepath'], mask=self.mask), self.mask
    
    def id2txt(self, id_list):
        """
        根据id获取对应的文本
        :params id_list id的列表[1,3,...]
        """
        for id in id_list:
            if id >= len(self.index2char):
                raise Exception("index out of the lengths请检查id的大小范围")
        return ''.join([self.index2char[id] for id in id_list])

    def __len__(self):
        return len(self.datasets)


"""
音频FBank特征提取
"""
class AudioParser:
    def __init__(self, win_len=0.02, sr=16000, hop_len=0.01):
        self.win_len = win_len
        self.sr = sr
        self.rect_masks = 5
        self.rect_freq = 50
        self.rect_time = 120
        self.rand = random.Random()
        win_bin = int(self.win_len * self.sr)
        hop_length = int(hop_len * self.sr)
        self.mel_transformer = torchaudio.transforms.MelSpectrogram(self.sr, n_fft=512, pad=0,
                                                                    win_length=win_bin,
                                                                    hop_length=hop_length, n_mels=64)
        self.amplitudeToDB = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.audio_f_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
        self.audio_t_mask = torchaudio.transforms.TimeMasking(time_mask_param=100)

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """
        对音频做cutout
        :param x: shape(1, 64, T)
        :return: tensor shape(1, 64, T)
        """
        sh = x.shape
        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                w_x = int(self.rand.uniform(0, self.rect_freq))
                w_y = int(self.rand.uniform(0, self.rect_time))

                rect_x = int(self.rand.uniform(0, sh[1] - w_x))
                rect_y = int(self.rand.uniform(0, sh[2] - w_y))

                mask[idx, rect_x: rect_x + w_x, rect_y: rect_y + w_y] = 1

        x = x.masked_fill(mask.type(torch.bool), 0)
        return x

    def spec_augment(self, x: torch.Tensor, freq_mask: Union[int, float] = 27,
                     time_mask: Union[int, float] = 100) -> torch.Tensor:
        """
        音频做augment
        :param time_mask: 时域上mask，可选int表示mask点，float表示mask比例
        :param freq_mask: 频域上mask, 可选int表示mask点，float表示mask比例
        :param x: x.shape = (1, 64, T)
        :return: (1, 64, T)
        """
        if isinstance(freq_mask, float):
            freq_mask = int(x.shape[1]*freq_mask)
        if isinstance(time_mask, float):
            time_mask = int(x.shape[2]*time_mask)

        sh = x.shape
        mask = torch.zeros(x.shape).byte()
        w_x = int(self.rand.uniform(0, freq_mask))
        w_y = int(self.rand.uniform(0, time_mask))

        rect_x = int(self.rand.uniform(0, sh[1] - w_x))
        rect_y = int(self.rand.uniform(0, sh[2] - w_y))

        mask[0, rect_x: rect_x + w_x, :] = 1
        mask[0, :, rect_y: rect_y + w_y] = 1
        x = x.masked_fill(mask.type(torch.bool), 0)
        return x

    def sample_aug(self, x: torch.Tensor, prob: float = 0.4) -> torch.Tensor:
        """
        随机丢失mel谱点
        :param x: (1, 64, T)
        :param prob: 概率
        :return: (1, 64, T)
        """
        prob = random.uniform(0., prob)
        mask = np.random.uniform(0, 0.5 / (1 - prob), size=x.shape)
        mask = np.round(mask)
        mask = torch.from_numpy(mask).byte()
        x = x.masked_fill(mask.type(torch.bool), 0)
        return x

    def sub_secquence(self, x: torch.Tensor, weight: float = 0.1):
        """
        获取子序列
        :param x: (T)
        :param weight:
        :return:
        """
        length = x.shape[1]
        target_length = int(length * np.random.uniform(weight, 1))  # 0-0.1随机采样
        location = int(np.random.uniform(0, length - target_length))
        return x[:, location:target_length]

    def parse_audio(self, audio_path, mask=False):
        if isinstance(audio_path, str) and not os.path.exists(path=audio_path):
            raise Exception("音频路径不存在 " + audio_path)
        y, sr = torchaudio.load(audio_path)
        # # dither
        y += 1e-5 * torch.randn_like(y)
        # do preemphasis
        y = torch.cat((y[:, 0].unsqueeze(1), y[:, 1:] - 0.97 * y[:, :-1]), dim=1, )
        if mask:
            y = self.sub_secquence(y, weight=0.98)
        spec = self.mel_transformer(y)
        y = self.amplitudeToDB(spec)
        # F-T mask
        if mask:
            # y = self.sample_aug(y, 0.2)
            y = self.spec_augment(y, freq_mask=27, time_mask=0.07)
            # y = self.audio_f_mask(y)
            # y = self.audio_t_mask(y)
            # cutout
            # y = self.cutout(y)
        # 归一化
        std, mean = torch.std_mean(y)
        y = torch.div((y - mean), std)

        return y  # (1,64,T)


class SSLDataModule(pl.LightningDataModule):
    def __init__(self, train_manifest: Union[ListConfig, str],
                 dev_manifest: Union[ListConfig, str],
                 test_manifest: Union[ListConfig, str],
                 pesudo_train_manifest: Union[ListConfig, str],
                 labels: list, train_bs=16, dev_bs=16, num_worker=0, 
                 ssl_model=None, ssl_model_retrain=False, on_the_flying:bool=True, ssl_folder=None):
        super().__init__()
        self.train_manifest = list(train_manifest) if isinstance(train_manifest, ListConfig) else [train_manifest]
        self.dev_manifest = list(dev_manifest) if isinstance(dev_manifest, ListConfig) else [dev_manifest]
        self.test_manifest = list(test_manifest) if isinstance(test_manifest, ListConfig) else [test_manifest]
        self.pesudo_train_manifest = list(pesudo_train_manifest) if isinstance(pesudo_train_manifest, ListConfig) else [pesudo_train_manifest]
        self.train_bs = train_bs
        self.dev_bs = dev_bs
        self.labels = labels
        self.num_worker = num_worker
        self.ssl_folder = ssl_folder
        self.on_the_flying = on_the_flying
        if on_the_flying and ssl_model is None:
            raise Exception("模型未初始化")
        self.ssl_model = ssl_model
        if ssl_model is not None and ssl_model_retrain:
            self.ssl_model.train()
        elif ssl_model is not None and ssl_model_retrain is False: self.ssl_model.eval()
        self.ssl_model_retrain = ssl_model_retrain
        self.rand = random.Random()
        self.pesudo_datasets = []
        self.origin_train_datasets = []


    def setup(self, stage=None):
        self.train_datasets = SSLAudioDataset(self.train_manifest, self.labels, mask=True,on_the_flying=self.on_the_flying, ssl_folder=self.ssl_folder)
        self.dev_datasets = SSLAudioDataset(self.dev_manifest, self.labels, max_duration=40,on_the_flying=self.on_the_flying, ssl_folder=self.ssl_folder)
        self.test_datasets = SSLAudioDataset(self.test_manifest, self.labels, max_duration=40,on_the_flying=self.on_the_flying, ssl_folder=self.ssl_folder)
        self.pesudo_train_datasets = SSLAudioDataset(self.pesudo_train_manifest, self.labels, max_duration=40,on_the_flying=self.on_the_flying, ssl_folder=self.ssl_folder)
        # self.train_datasets.datasets.extend(self.pesudo_datasets)
        self.origin_train_datasets = self.train_datasets.datasets
        print("训练数据集大小: {:d}".format(len(self.train_datasets)))
        logging.info("训练数据集大小: {:d}".format(len(self.train_datasets)))


    def train_dataloader(self):
        self.train_datasets.datasets = self.origin_train_datasets + self.pesudo_datasets
        self.pesudo_datasets.clear()
        logging.info("训练数据集大小: {:d}".format(len(self.train_datasets)))
        return DataLoader(self.train_datasets, batch_size=self.train_bs, num_workers=self.num_worker,
                          pin_memory=True, collate_fn=self._collate_fn, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_datasets, batch_size=self.dev_bs, num_workers=self.num_worker,
                          pin_memory=True, collate_fn=self._collate_fn, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_datasets, batch_size=self.dev_bs, num_workers=self.num_worker,
                          pin_memory=True, collate_fn=self._collate_fn)
    
    def inject_pesudo_datasets(self, datas):
        """
        注入伪标签数据
        :params
            datas: [(audio_path, text),....]
        """
        for audio_path, text in datas:
            if os.path.exists(audio_path) and self.ssl_folder is not None:
                ssl_audio_path = os.path.join(self.ssl_folder, audio_path.split('/')[-1].split('.wav')[0]+'.pkl')
                self.pesudo_datasets.append({"ssl_audio_filepath": ssl_audio_path,"audio_filepath":audio_path, "text": text})
            if os.path.exists(audio_path) and self.ssl_folder is None:
                self.pesudo_datasets.append({"audio_filepath": audio_path, "text": text})
            if not os.path.exists(audio_path):
                logging.warning("路径不存在{:s}".format(audio_path))
    def pseudo_train_dataloader(self):
        """
        用于自监督训练的无标注数据集
        """
        return DataLoader(self.pesudo_train_datasets, batch_size=self.dev_bs, num_workers=self.num_worker,
                          pin_memory=True, collate_fn=self._collate_fn, shuffle=True)

    def get_train_step(self):
        """
        获取每轮的step大小
        :return: steps
        """
        return len(self.train_dataloader())

    def _collate_fn(self, batch):
        """
        特征批处理，或者离线加载, 数据主要由两部分组成，mel数据和wav2vec数据
        """
        minibatch_size = len(batch)
        # mel域数据
        longest_sample = max(batch, key=lambda x: x[2].size(2))[2]  # (1,64,T)
        freq_size = longest_sample.size(1)
        max_seqlength = longest_sample.size(2)  # 时域长度
        mel_inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)  # mel输入
        # 文本数据
        max_trans_length = len(max(batch, key=lambda x: len(x[1]))[1])  # 文本长度
        # wav2vec数据
        wav2vec_inputs = None
        input_percentages = None
        if self.on_the_flying and not self.ssl_model_retrain:  # 如果自监督模型不需要重新训练
            with torch.no_grad():
                wav2vec_inputs, input_percentages = self.ssl_model([item[0] for item in batch])
                wav2vec_inputs = wav2vec_inputs.unsqueeze(1)  # B*1*T*512
        if self.on_the_flying and self.ssl_model_retrain:  # 如果模型需要重新训练
            wav2vec_inputs, input_percentages = self.ssl_model([item[0] for item in batch])
            wav2vec_inputs = wav2vec_inputs.unsqueeze(1)
        if not self.on_the_flying:
            # 如果使用文件夹加载特征
            audio_lists = []
            for i in range(len(batch)):
                fp = open(batch[i][0], 'rb')
                audio_lists.append(pickle.load(fp))  # append np.array 1*T*512
                fp.close()
            # 构建输入向量
            max_seq_len = max(audio_lists, key=lambda x:x.shape[1]).shape[1]
            feature_dim = max(audio_lists, key=lambda x:x.shape[1]).shape[2]
            wav2vec_inputs = torch.zeros(minibatch_size, 1, max_seq_len, feature_dim)
            input_percentages = torch.FloatTensor(minibatch_size)  # mask 使用
            for i in range(len(batch)):
                wav2vec_inputs[i].narrow(1, 0,audio_lists[i].shape[1]).copy_(torch.as_tensor(data=audio_lists[i]))
                input_percentages[i] = audio_lists[i].shape[1] / float(max_seq_len)
        wav2vec_inputs = wav2vec_inputs.transpose(2,3)  # B*1*E*T
        wav2vec_inputs = wav2vec_inputs.cpu()
        if batch[0][3]:  # 是否mask，测试集不mask
            wav2vec_inputs = self.features_cutout(wav2vec_inputs)
        target_sizes = torch.IntTensor(minibatch_size)
        targets = torch.zeros(minibatch_size, max_trans_length)
        paths = []
        for x in range(minibatch_size):
            sample = batch[x]
            # mel数据组装
            tensor = sample[2].squeeze(0)  # (64,T)
            seq_length = tensor.size(1)  # 时域长度T
            mel_inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            # 文本数据
            trans_txt = sample[1]
            audio_path = sample[2]  # 音频路径
            paths.append(audio_path)
            target_sizes[x] = len(trans_txt)
            targets[x].narrow(0, 0, len(trans_txt)).copy_(torch.IntTensor(trans_txt))
        targets = targets.long()  # label 文本
        return (wav2vec_inputs, mel_inputs), targets, input_percentages, target_sizes, paths

    def features_cutout(self, x:torch.Tensor) -> torch.Tensor:
        """
        对特征做随机cutout裁剪 B*1*E*T
        """
        sh = x.shape
        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(5):
                w_x = int(self.rand.uniform(0, 150))
                w_y = int(self.rand.uniform(0, 100))

                rect_x = int(self.rand.uniform(0, sh[2] - w_x))
                rect_y = int(self.rand.uniform(0, sh[3] - w_y))

                mask[idx, 0, rect_x: rect_x + w_x, rect_y: rect_y + w_y] = 1

        x = x.masked_fill(mask.type(torch.bool), 0)
        return x

@hydra.main(config_path='../conf', config_name='conf')
def main(cfg: DictConfig):
    data_cfg = cfg.get('data')
    ssl_model = Wav2Vec2Extractor(model_path="/data/chenc/asr/lightning-asr/ckpt/ckpt", device="cuda:0")
    dataset = SSLAudioDataset(data_cfg.get('train_manifest'), data_cfg.get('labels'))
    # datasets测试
    data = dataset.__getitem__(1)
    txt2id = data[1]
    id2txt = dataset.id2txt(txt2id)
    labels = [c.strip() for c in open("/data/chenc/aishell/vocab.txt", 'r').readlines()]
    # dataloader 测试
    dataloader = SSLDataModule("/data/chenc/aishell/test.json", "/data/chenc/aishell/test.json",
                                    test_manifest="/data/chenc/aishell/test.json", labels=labels, ssl_model=ssl_model,
                                    train_bs=4, dev_bs=4, ssl_folder="/data/chenc/aishell/ssl/wav2vec2", on_the_flying=False,
                                    pesudo_train_manifest="/data/chenc/aishell/test.json")
    dataloader.setup()
    for batch in enumerate(dataloader.train_dataloader()):
        print("inputs:" + str(batch[1][0][0].size()))
        print("targets:" + str(batch[1][1].size()))
        print("input_percentages:" + str(batch[1][2].size()))
        print("target_sizes:" + str(batch[1][3].size()))
        print(len(batch[1][4]))


if __name__ == '__main__':
    main()
