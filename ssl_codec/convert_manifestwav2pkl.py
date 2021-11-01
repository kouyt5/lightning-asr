import logging
import pickle
import json, os
from typing import List, Tuple, Union
import torch
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2ForCTC,Wav2Vec2Model
import soundfile as sf
import time
import numpy as np
import scipy.signal
from tqdm import tqdm


class Model:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, audios: Union[str, List[str]]) -> Tuple[torch.Tensor,torch.Tensor]:
        pass


class Wav2Vec2Extractor(torch.nn.Module):
    def __init__(self, model_path:str = "facebook/wav2vec2-large-xlsr-53",
                        device:str = "cpu",
                        target_sample_rate:int = 16000) -> None:
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, cache_dir=".ckpt") #, map_location="cpu"
        if not device == "cpu" and not torch.cuda.is_available():
            logging.warning("gpu 不可用，使用cpu推理")
            device = "cpu"
        self.device = torch.device(device)
        self.model = Wav2Vec2Model.from_pretrained(model_path, cache_dir=".ckpt").to(self.device)
    
    def freeze(self):
        for name,param in self.model.named_parameters():
            param.requires_grad=False

    def forward(self, audios: Union[str, List[str]]) -> Tuple[torch.Tensor,torch.Tensor]:
        datas = []
        if isinstance(audios, str):
            datas, sr = sf.read(audios)
        else:
            for audio in audios:
                data, sr = sf.read(audio)
                if not sr == self.target_sample_rate:  # 针对采样率有问题的音频
                    logging.warning("音频{:s}采样率非16k，降采样...".format(audio))
                    # resample
                    target_samples = int(float(len(data)/sr)*self.target_sample_rate)
                    data = scipy.signal.resample(data, target_samples)
                datas.append(data)
        feature = self.feature_extractor(datas, return_tensors="pt", padding=True, sampling_rate=self.target_sample_rate).input_values.to(self.model.device)
        out = self.model(feature).extract_features
        # 选择最长的作为1，返回百分比，用于模型知道哪些被pad了
        longest_sample_size = max(datas, key=lambda x: x.shape[0]).shape[0]  # 16000
        input_percentages = torch.FloatTensor(len(datas))
        for i in range(len(datas)):
            input_percentages[i] = float(datas[i].shape[0])/longest_sample_size
        return out, input_percentages



def main():
    model = Wav2Vec2Extractor(model_path="/data/chenc/aishell/ssl/model", device="cpu")
    pre_time = time.time()
    # out = model("/data/chenc/asr/lightning-asr/test.wav")
    # out = model("/data/chenc/asr/lightning-asr/test.wav")
    print(time.time()-pre_time)
    pre_time = time.time()
    out = model(["/data/chenc/asr/lightning-asr/test.wav","/data/chenc/asr/lightning-asr/test2.wav"])
    print(time.time()-pre_time)
    
@torch.no_grad()
def convert(source_path:str, target_path:str, model:Model): 
    """
    将一个音频通过无监督算法提取其特征，用于替换FBank
    """
    tensor_data = model([source_path])[0]
    np_data = tensor_data.detach().cpu().numpy()
    pickle.dump(np_data,open(os.path.join(target_path, source_path.split('/')[-1].split('.wav')[0]+'.pkl'), mode='wb'), 1)

def convert_manifest(file:str, target_path:str, model=None):
    with open(file=file) as f:
        all_lines = f.readlines()
        for line in tqdm(all_lines):
            audio_path = json.loads(line)['audio_filepath']
            convert(source_path=audio_path, target_path=target_path, model=model)
if __name__ == '__main__':
    # model = Wav2Vec2Extractor(model_path="../ckpt/ckpt", device="cuda:0")
    # convert_manifest("/data/chenc/aishell/train.json","/data/chenc/aishell/ssl/wav2vec2", model=model)
    main()
