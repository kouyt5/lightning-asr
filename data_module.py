import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import os
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
import random


class MyAudioDataset(Dataset):
    def __init__(self, manifest_path, labels, max_duration=16, mask=False, win_len=0.02, sr=16000):
        torchaudio.set_audio_backend("sox_io")
        self.mask = mask
        self.win_len = win_len
        self.sr = sr
        self.datasets = []
        self.labels = labels
        # cutout
        self.rect_masks = 5
        self.rect_freq = 50
        self.rect_time = 120
        self.rand = random.Random()
        with open(manifest_path, encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line, encoding='utf-8')
                if data['duration'] > max_duration:
                    continue
                self.datasets.append(data)
        self.index2char = dict([(i, labels[i]) for i in range(len(labels))])
        self.char2index = dict([(labels[i], i) for i in range(len(labels))])
        win_bin = int(self.win_len * self.sr)
        hop_length = win_bin // 2
        self.mel_transformer = torchaudio.transforms.MelSpectrogram(self.sr, n_fft=512, pad=16,
                                                                    win_length=win_bin,
                                                                    hop_length=hop_length, n_mels=64)
        self.amplitudeToDB = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.audio_f_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
        self.audio_t_mask = torchaudio.transforms.TimeMasking(time_mask_param=100)


    def __getitem__(self, index):
        data = self.datasets[index]
        text2id = [self.char2index[char] for char in data['text']]
        return self.parse_audio(data["audio_filepath"], mask=self.mask), text2id, data['audio_filepath']

    def cutout(self, x:torch.Tensor) -> torch.Tensor:
        """
        对音频做cutout
        :param x: shape(1, 64, T)
        :return: tensor shape(1, 64, T)
        """
        sh = x.shape
        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = int(self.rand.uniform(0, sh[1] - self.rect_freq))
                rect_y = int(self.rand.uniform(0, sh[2] - self.rect_time))

                w_x = int(self.rand.uniform(0, self.rect_freq))
                w_y = int(self.rand.uniform(0, self.rect_time))

                mask[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 1

        x = x.masked_fill(mask.type(torch.bool), 0)
        return x

    def parse_audio(self, audio_path, mask=False):
        if not os.path.exists(path=audio_path):
            raise ("音频路径不存在 " + audio_path)
        y, sr = torchaudio.load(audio_path, self.sr)
        # # dither
        y += 1e-5 * torch.randn_like(y)
        # do preemphasis
        y = torch.cat((y[:, 0].unsqueeze(1), y[:, 1:] - 0.97 * y[:, :-1]), dim=1,)

        spec = self.mel_transformer(y)
        y = self.amplitudeToDB(spec)
        # F-T mask
        if mask:
            # y = self.audio_f_mask(y)
            # y = self.audio_t_mask(y)
            # cutout
            y = self.cutout(y)
        # 归一化
        std, mean = torch.std_mean(y)
        y = torch.div((y - mean), std)

        return y  # (1,64,T)


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


class LibriDataModule(pl.LightningDataModule):
    def __init__(self, train_manifest: str, dev_manifest: str, labels: list, train_bs=16, dev_bs=16):
        super().__init__()
        self.train_manifest = train_manifest
        self.dev_manifest = dev_manifest
        self.train_bs = train_bs
        self.dev_bs = dev_bs
        self.labels = labels

    def setup(self, stage=None):
        self.train_datasets = MyAudioDataset(self.train_manifest, self.labels, mask=True)
        self.dev_datasets = MyAudioDataset(self.dev_manifest, self.labels, max_duration=40)
        self.test_datasets = MyAudioDataset(self.dev_manifest, self.labels, max_duration=40)

    def train_dataloader(self):
        return DataLoader(self.train_datasets, batch_size=self.train_bs, num_workers=6,
                          pin_memory=True, collate_fn=self._collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dev_datasets, batch_size=self.dev_bs, num_workers=6,
                          pin_memory=True, collate_fn=self._collate_fn, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.dev_datasets, batch_size=self.dev_bs, num_workers=6,
                          pin_memory=True, collate_fn=self._collate_fn)

    def get_train_step(self):
        """
        获取每轮的step大小
        :return: steps
        """
        return len(self.train_dataloader())
    # def transfer_batch_to_device(self, batch, device):
    #     batch[0][0] = batch[0][0].to(device)
    #     batch[0][2] = batch[0][2].to(device)
    #     return batch
    def _collate_fn(self, batch):
        # batch = sorted(  # batch_size * {(1,64,T),[1,4,...],duration}
        #     batch, key=lambda sample: sample[0].size(2), reverse=True)
        longest_sample = max(batch, key=lambda x: x[0].size(2))[0]  # (1,64,T)
        freq_size = longest_sample.size(1)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(2)  # 时域长度
        max_trans_length = len(max(batch, key=lambda x: len(x[1]))[1])  # 文本长度
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size)  # mask 使用
        target_sizes = torch.IntTensor(minibatch_size)  #
        # targets = torch.zeros(minibatch_size,max_trans_length,dtype=torch.int16)
        targets = torch.zeros(minibatch_size, max_trans_length)
        paths = []
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0].squeeze(0)  # (64,T)
            trans_txt = sample[1]
            audio_path = sample[2]
            paths.append(audio_path)
            seq_length = tensor.size(1)  # 时域长度T
            inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[x] = seq_length / float(max_seqlength)
            target_sizes[x] = len(trans_txt)
            targets[x].narrow(0,0,len(trans_txt)).copy_(torch.IntTensor(trans_txt))
        targets = targets.long()
        return inputs, targets, input_percentages, target_sizes, paths


@hydra.main(config_path='conf', config_name='conf')
def main(cfg: DictConfig):
    data_cfg = cfg.get('data')
    dataset = MyAudioDataset(data_cfg.get('train_manifest'), data_cfg.get('labels'), mask=True)
    # datasets测试
    data = dataset.__getitem__(1)
    txt2id = data[1]
    id2txt = dataset.id2txt(txt2id)
    # dataloader 测试
    dataloader = LibriDataModule(data_cfg.get('train_manifest'), data_cfg.get('val_manifest'),
                                  labels=data_cfg.get('labels')).train_dataloader()
    for batch in enumerate(dataloader):
        print("inputs:" + str(batch[1][0].size()))
        print("targets:" + str(batch[1][1].size()))
        print("input_percentages:" + str(batch[1][2].size()))
        print("target_sizes:" + str(batch[1][3].size()))
        print(len(batch[1][4]))


if __name__ == "__main__":
    main()
