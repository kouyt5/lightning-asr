import torch
import numpy as np

from ctc_decoders import Scorer, ctc_beam_search_decoder_batch
"""
# 安装语言模型
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/NVIDIA/OpenSeq2Seq -b ctc-decoders
mv OpenSeq2Seq/decoders .
rm -rf OpenSeq2Seq
cd decoders
./setup.sh
cd ..
"""


class BeamSearchDecoderWithLM(torch.nn.Module):

    def __init__(
        self, vocab, beam_width, alpha, beta, lm_path, num_cpus, cutoff_prob=1.0, cutoff_top_n=40):

        if lm_path is not None:
            self.scorer = Scorer(alpha, beta, model_path=lm_path, vocabulary=vocab)
        else:
            self.scorer = None
        self.vocab = vocab
        self.beam_width = beam_width
        self.num_cpus = num_cpus
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        probs = self.revert_softmax(log_probs)
        probs_list = []
        for i, prob in enumerate(probs):
            probs_list.append(prob[: log_probs_length[i], :])
        results = ctc_beam_search_decoder_batch(
            probs_list,
            self.vocab,
            beam_size=self.beam_width,
            num_processes=self.num_cpus,
            ext_scoring_func=self.scorer,
            cutoff_prob=self.cutoff_prob,
            cutoff_top_n=self.cutoff_top_n,
        )
        result = [item[0][1] for item in results]
        return result

    def revert_softmax(self, logits):
        """
        对对数概率还原其softmax值，用于计算语言模型分数
        """
        result = np.zeros_like(logits)
        for i in range(logits.shape[0]):
            item = logits[i]
            e = np.exp(item - np.max(item))
            result[i] = e / e.sum(axis=-1).reshape([item.shape[0], 1])
        return result

if __name__ == '__main__':
    vocab = [c.strip() for c in open("data/aishell1-vocab.txt", 'r').readlines()]
    lm_path = "/data/chenc/asr/minhang/atc-service/asr/checkpoints/kenlm/cn.arpa"
    decoder = BeamSearchDecoderWithLM(vocab=vocab,
                            beam_width=40,
                            alpha=1.,
                            beta=1.,
                            lm_path=lm_path,
                            num_cpus=6,
                            cutoff_prob=1, cutoff_top_n=40)
    log_prob = torch.randn((2,1000,4334), dtype=torch.float32)
    log_prob = torch.log_softmax(log_prob, dim=-1).numpy()
    lengths = torch.IntTensor([100,200]).numpy()
    
    out = decoder.forward(log_probs=log_prob, log_probs_length=lengths)
    print()

    