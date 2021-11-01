from numpy import mod
from beam_search import BeamSearchDecoderWithLM
from ssl_codec.convert_manifestwav2pkl import Wav2Vec2Extractor
from ssl_codec.ssl_data_module import SSLDataModule
from ssl_codec.utils import sum_logprob
from train import LightingModule
import pytorch_lightning as pl
from data_module import AudioParser
import torch
from train_ssl import SSLLightingModule
from utils.asr_metrics import WER, word_error_rate
import time
from data_module import LibriDataModule
import io, os
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
        self.model.eval()

    @torch.no_grad()
    def translate(self, audio_path:str):
        """
        将一个本地音频转录为文本
        :param
            audio_path: 音频路径
        :return
            text: 文本
        """
        pre_time = time.time()
        audio_tensor = self.audio_parser.parse_audio(audio_path=audio_path, mask=False)
        print("加载音频用时: "+str(time.time()-pre_time))
        pre_time = time.time()
        model_in = (audio_tensor.to(self.device), torch.FloatTensor([1.]).to(self.device))
        model_out = self.model.forward(*model_in)
        print("模型计算用时: "+str(time.time()-pre_time))
        pre_time = time.time()
        text = self.wer.ctc_decoder_predictions_tensor(torch.argmax(model_out, dim=-1, keepdim=False))[0]
        print("解码用时: "+str(time.time()-pre_time))
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

class AsrTranslatorSSL:
    def __init__(self, model_path:str, map_location="cpu", lang:str="en", 
                    lable_path:str=None, use_lm:bool = False,
                    lm_model:torch.nn.Module = None,
                    ssl_model:Wav2Vec2Extractor = None):
        """
        语音识别模型测试
        param:
            model_path: 模型路径
            map_location: 使用gpu还是cpu, "cuda:0" or "cpu"
            ssl_model: 自监督模型
        """
        # cn
        if lang == "en":
            self.labels = [" ","'","a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        else:
            self.labels = [c.strip() for c in open(lable_path, 'r').readlines()]
        # 语言模型
        self.lm_model = lm_model
        self.use_lm = use_lm
        if use_lm and lm_model is None:
            raise Exception("语言模型未初始化，如果不使用语言模型，请将use_lm参数置为False")
        self.wer = WER(vocabulary=self.labels, use_cer=True)
        self.model_path = model_path  # 用于trainer
        self.map_location = map_location
        self.model = SSLLightingModule.load_from_checkpoint(model_path, map_location=map_location)
        self.device = torch.device(map_location)
        self.audio_parser = ssl_model.to(self.device)
        self.model = self.model.to(self.device)
        self.wer = WER(vocabulary=self.labels)
        self.model.eval()

    @torch.no_grad()
    def translate(self, audio_path:str):
        """
        将一个本地音频转录为文本
        :param
            audio_path: 音频路径
        :return
            text: 文本
        """
        pre_time = time.time()
        audio_tensor, percents = self.audio_parser([audio_path])
        # print("加载音频用时: "+str(time.time()-pre_time))
        pre_time = time.time()
        model_in = (audio_tensor.unsqueeze(1).transpose(2,3).to(self.device), torch.FloatTensor([1.]).to(self.device))
        model_out = self.model.forward(*model_in)
        t_lengths = torch.mul(model_out.size(1), percents).int()
        avg_prob = sum_logprob(model_out, t_lengths)
        # print("模型计算用时: "+str(time.time()-pre_time))
        pre_time = time.time()
        if self.use_lm:
            text = self.lm_model.forward(log_probs=model_out.cpu().numpy(), log_probs_length=t_lengths)[0]
        else:
            text = self.wer.ctc_decoder_predictions_tensor(torch.argmax(model_out, dim=-1, keepdim=False))[0]
        # print("解码用时: "+str(time.time()-pre_time))
        return text, avg_prob
    
    @torch.no_grad()
    def evalute_manifest(self, test_manifest:str, batch_size=32, num_workers=6, ssl_model=None, ssl_folder=None):
        """
        测试一个manifest.json中的准确率
        :param
            test_manifest: 测试集的manifest文件
        """
        from tqdm import tqdm
        data_module = SSLDataModule(train_manifest=test_manifest,dev_manifest=test_manifest,
                                    test_manifest=test_manifest, pesudo_train_manifest=test_manifest,
                                    dev_bs=batch_size, num_worker=num_workers,
                                    labels=self.labels, ssl_model=None, ssl_folder=ssl_folder, on_the_flying=False)
        trainer = pl.Trainer(gpus=0 if self.map_location=="cpu" else 1)
        trainer.test(self.model, datamodule=data_module)
        # 开始语言模型解码
        data_module.setup(stage="test")
        total_count = 0.
        total_cer = 0.
        for i, batch in enumerate(tqdm(data_module.test_dataloader())):
            batch  # inputs, targets, input_percentages, target_sizes, paths
            inputs = batch[0]
            percentage = batch[2]
            model_out = self.model.forward(inputs.to(self.device), percentage.to(self.device))
            t_lengths = torch.mul(model_out.size(1), batch[2]).int()
            texts = self.lm_model.forward(model_out.cpu().numpy(), t_lengths)
            trues = self.wer.decode_reference(batch[1], target_lengths=batch[3])
            wer = word_error_rate(texts, references=trues, use_cer=True)
            total_count += 1
            total_cer += wer
        print("语言模型平均cer为:{:.4f}".format(total_cer/total_count))
    
    def statistic_manifest_wer_by_prob(self, test_manifest:str):
        import json, csv
        from tqdm import tqdm
        all_result = []
        count = 0
        with open(test_manifest, mode='r') as f:
            for line in tqdm(f.readlines()):
                predict_text, prob = self.translate(json.loads(line)['audio_filepath'])
                ground_trues_text = json.loads(line)['text']
                cer = word_error_rate([predict_text], [ground_trues_text], use_cer=True)
                all_result.append({
                "path": json.loads(line)['audio_filepath'],
                "pred": predict_text,
                "true": ground_trues_text,
                "cer": cer,
                'prob': prob[0]
            })
                count += 1
                if count > 10000:
                    break
        writer = csv.DictWriter(open('result.csv', 'w'), fieldnames=['path', 'pred', 'true', 'cer', 'prob'])
        writer.writeheader()
        for item in all_result:
            writer.writerow(item)


def main_translator():
    model_path = "/data/chenc/asr/lightning-asr/outputs/asr13x1_x/2021-08-20/21-04-55/checkpoints/last.ckpt"
    asr_translator = AsrTranslator(model_path=model_path, map_location="cuda:0", lang="en")
    audio_path = "/data/chenc/libri/train-clean-100-processed/103-1240-0010.wav"

    # 用于byte类数据测试，等效于路径，方便在线输入
    byte_io = io.BytesIO(io.FileIO(audio_path).read())
    pre_time = time.time()
    text = asr_translator.translate(byte_io)
    # text = asr_translator.translate(audio_path=audio_path)
    print("转录用时: " + str(time.time()-pre_time))

    pre_time = time.time()
    text = asr_translator.translate(audio_path=audio_path)
    print("转录用时: " + str(time.time()-pre_time))
    print("转录结果为: " + text)
    print("开始测试")
    test_manifest = "/data/chenc/libri/dev-clean.json"
    asr_translator.evalute_manifest(test_manifest=test_manifest)
def main_ssl():
    ssl_model = Wav2Vec2Extractor(model_path="ckpt/ckpt", device="cuda:0")
    asr_model_path = "/data/chenc/asr/lightning-asr/outputs/aishell-low-asr13x1-ssl/2021-11-01/12-22-aishell-low-asr13x1-ssl-lr0.05-wc0.001-bs32-gpus1-dr0.2-mask_True/checkpoints/asr-epoch=99-val_wer=0.30.ckpt"
    lable_path = "/data/chenc/asr/lightning-asr/data/aishell1-vocab.txt"
    vocab = [c.strip() for c in open("data/aishell1-vocab.txt", 'r').readlines()]
    lm_path = "/data/chenc/asr/lightning-asr/ckpt/lm/2.arpa"
    decoder = BeamSearchDecoderWithLM(vocab=vocab,
                            beam_width=40,
                            alpha=1.,
                            beta=1.,
                            lm_path=lm_path,
                            num_cpus=os.cpu_count(),
                            cutoff_prob=1, cutoff_top_n=40)
    asr_translator = AsrTranslatorSSL(model_path=asr_model_path, map_location="cuda:0", 
                                        lang='cn', lable_path=lable_path, ssl_model=ssl_model,
                                        use_lm=True, lm_model=decoder)
    text, prob = asr_translator.translate("/data/chenc/asr/lightning-asr/test.wav")
    print(prob)
    test_manifest = "/data/chenc/aishell/test.json"
    asr_translator.evalute_manifest(test_manifest=test_manifest, ssl_model=ssl_model, ssl_folder="/data/chenc/aishell/ssl/wav2vec2")
    # asr_translator.statistic_manifest_wer_by_prob(test_manifest)

    
    print()
if __name__ == "__main__":
    main_ssl()
