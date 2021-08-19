from numpy import mod
from train import LightingModule
import pytorch_lightning as pl
from data_module import AudioParser
import torch
from utils.asr_metrics import WER
import time
from data_module import LibriDataModule
import io
import logging


logging.basicConfig(level=logging.DEBUG)


class AsrTranslator:
    def __init__(self, model_path:str, map_location="cpu", lang:str="en"):
        """
        语音识别模型测试
        param:
            model_path: 模型路径
            map_location: 使用gpu还是cpu, "cuda:0" or "cpu"
        """
        # cn
        if lang == "en":
            self.labels = [" ","'","a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        else:
            raise Exception("其他语言未实现")
        self.model_path = model_path  # 用于trainer
        self.map_location = map_location
        self.model = LightingModule.load_from_checkpoint(model_path, map_location=map_location)
        self.audio_parser = AudioParser()
        self.device = self.model.device
        self.wer = WER(vocabulary=self.labels)

    @torch.no_grad()
    def translate(self, audio_path:str):
        """
        将一个本地音频转录为文本
        :param
            audio_path: 音频路径
        :return
            text: 文本
        """
        audio_tensor = self.audio_parser.parse_audio(audio_path=audio_path, mask=False)
        model_in = (audio_tensor.to(self.device), torch.FloatTensor([1.]).to(self.device))
        model_out = self.model.forward(*model_in)
        text = self.wer.ctc_decoder_predictions_tensor(torch.argmax(model_out, dim=-1, keepdim=False))[0]
        return text
    
    def evalute_manifest(self, test_manifest:str, batch_size=32, num_workers=6):
        """
        测试一个manifest.json中的准确率
        :param
            test_manifest: 测试集的manifest文件
        """
        data_module = LibriDataModule(train_manifest=test_manifest,dev_manifest=test_manifest,
                                    test_manifest=test_manifest, dev_bs=batch_size, num_worker=num_workers,
                                    labels=self.labels)
        trainer = pl.Trainer(gpus=0 if self.map_location=="cpu" else 1)
        trainer.test(self.model, datamodule=data_module)


if __name__ == "__main__":
    model_path = "/data/chenc/asr/lightning-asr/outputs/asr13x1-pad32/2021-08-19/16-39-05/checkpoints/last.ckpt"
    asr_translator = AsrTranslator(model_path=model_path, map_location="cuda:0", lang="en")
    audio_path = "/data/chenc/libri/train-clean-100-processed/103-1240-0010.wav"

    # 用于byte类数据测试，等效于路径，方便在线输入
    byte_io = io.BytesIO(io.FileIO(audio_path).read())
    pre_time = time.time()
    text = asr_translator.translate(byte_io)
    # text = asr_translator.translate(audio_path=audio_path)
    print("转录用时: " + str(time.time()-pre_time))
    print("转录结果为: " + text)
    print("开始测试")
    test_manifest = "/data/chenc/libri/dev-clean.json"
    asr_translator.evalute_manifest(test_manifest=test_manifest)
