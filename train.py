from typing import List, Any, Dict
from comet_ml import Experiment  # 必须引入，不然报错
from exp_loggers import init_loggers
import torch
import pytorch_lightning as pl
from data_module import LibriDataModule
from torch.utils.data import DataLoader
from scheduler.cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import get_mnist_pair
from models.QuartNet import MyModel2
from decoder import GreedyDecoder
from utils.asr_metrics import WER
import logging
logger = logging.getLogger(__name__)


class LightingModule(pl.LightningModule):

    def forward(self, x):
        """
        用于推理阶段

        :param x: 传入参数 N*1*64×L
        :return: lists 预测的原始值
        """
        return self.encoder(x)  # N*L'*C

    def configure_optimizers(self):
        """
        配置优化器

        :return: torch.optim
        """
        self.print('设置学习率' + str(self.learning_rate))
        sgd_optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # adam_optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # lr_schulder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(sgd_optim, T_0=5,
        #                                                                    T_mult=2, last_epoch=-1)
        # lr_schulder = CosineAnnealingWarmupRestarts(sgd_optim, first_cycle_steps=5000, cycle_mult=2,
        #                                             max_lr=self.learning_rate, min_lr=1e-3, warmup_steps=1000, gamma=0.5)
        # lr_schulder = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optim, mode='min',
        #                                                          factor=0.2, patience=4,
        #                                                          threshold=1e-4, threshold_mode='rel',
        #                                                          min_lr=1e-4)
        lr_schulder = torch.optim.lr_scheduler.ExponentialLR(sgd_optim, gamma=0.96)
        pack_schulder = {
            'scheduler': lr_schulder,
            'interval': 'epoch',
            'monitor': 'val_loss',
        }
        return [sgd_optim], [pack_schulder]

    def training_step(self, batch, batch_idx):
        """
        每一步的训练，相当于原生pytorch的

        >>> for i, batch in enumerate(DataLoader()):
        >>>     self.training_step(batch, i)

        :param batch: 包含每一batch的数据
        :param batch_idx: index
        :return: loss: torch.Tensor, 或者一个字典, 但必须包含'loss' key
        """
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = self.encoder(input, percents)
        t_lengths = torch.mul(out.size(1), percents).int()  # 输出实际长度
        loss = self.loss(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_wer', self.wer(out.argmax(dim=-1, keepdim=False), trans, trans_lengths),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx%50 == 0:
            logging.info("pred:"+str(self.decoder.decode(out)[0][0]))
            logging.info("true:"+str(self.decoder.convert_to_strings(trans, remove_repetitions=False)[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        print(input.device, percents.device)
        out = self.encoder(input, percents)
        t_lengths = torch.mul(out.size(1), percents).int()  # 输出实际长度
        loss = self.loss(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths)
        self.log('val_wer', self.wer(out.argmax(dim=-1, keepdim=False), trans, trans_lengths),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return_val = {
            'val_loss': loss,
            'input': batch[0],
            'val_wer': self.wer(out.argmax(dim=-1, keepdim=False), trans, trans_lengths),
            'pred': self.decoder.decode(out),
            'true': self.decoder.convert_to_strings(trans, remove_repetitions=False)
        }
        return return_val

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        验证一轮后调用，用于总结总的验证集效果

        :param outputs: 每一step的结果的列表
        :return: None
        """
        pass
        # 找出一轮中准确率最低的batch的前8个
        # min_sample = min(outputs, key=lambda x: x['val_acc'])
        # grid = torchvision.utils.make_grid(min_sample['images'][0:8], nrow=4)
        # transform = torchvision.transforms.ToPILImage()
        # image = transform(grid)
        # pred = min_sample['pred'][0:8]
        # true = min_sample['true'][0:8]
        # 使用comet的logger
        # self.logger[1].experiment.log_image(image_data=image, name=str(pred)+str(true)+'.png')

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        保存checkpoint中调用，这里可以查看保存的字典

        :param checkpoint: 状态字典

        .. code-block:: python
            字典keys = ['epoch', 'global_step', 'pytorch-lightning_version',
            'callbacks', 'optimizer_states', 'lr_schedulers',
            'state_dict', 'hparams_name', 'hyper_parameters']

        :return: None
        """
        # logger.info(checkpoint.keys())
        pass

    def __init__(self, learning_rate=5e-3, weight_decay=1e-4, labels=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.save_hyperparameters()
        self.wer = WER(vocabulary=self.labels)
        self.loss = torch.nn.CTCLoss(blank=len(self.labels))  # 最后一个作为black
        self.encoder = MyModel2(labels=self.labels)
        self.decoder = GreedyDecoder(labels=self.labels)


@hydra.main(config_path='conf', config_name='conf')
def main(cfg: DictConfig):
    pl.seed_everything(0)  # 设置随机数seed
    print(OmegaConf.to_yaml(cfg=cfg))
    print(os.getcwd())
    tran_cfg = cfg.get('train')
    logger_cfg = cfg.get('loggers')
    data_cfg = cfg.get('data')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", monitor='val_loss', verbose=True,
                                                       save_last=True, save_top_k=3,
                                                       filename="mnist-{epoch:02d}-{val_loss:.2f}")
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    loggers = init_loggers(cfg=logger_cfg)
    data_module = LibriDataModule(data_cfg.get('train_manifest'), data_cfg.get('val_manifest'),
                                  labels=data_cfg.get('labels'), train_bs=tran_cfg.get('train_batch_size'),
                                  dev_bs=tran_cfg.get('dev_batch_size'))
    model = LightingModule(learning_rate=tran_cfg.get("learning_rate"),
                           weight_decay=tran_cfg.get("weight_decay"),
                           labels=data_cfg.get('labels'))
    trainer = pl.Trainer(gpus=tran_cfg.get('gpus'),
                            tpu_cores=tran_cfg.get('tpu_core_num'),  # google tpu训练
                            logger=loggers,
                            callbacks=[checkpoint_callback, lr_callback],
                            resume_from_checkpoint=tran_cfg.get('checkpoint'),
                            # auto_lr_find=True,
                            accelerator=tran_cfg.get("accelerator"),  # ddp
                            amp_level=tran_cfg.get('amp_level'),  # 'O1', 'O2'
                            precision=tran_cfg.get('precision'),  # 16 or 32
                            amp_backend=tran_cfg.get('amp_backend'),  # native推荐
                            profiler="simple",  # 打印各个函数执行时间
                            accumulate_grad_batches=1,  # 提高batch_size的办法
                            # limit_val_batches=0.02,
                            max_epochs=tran_cfg.get('total_epoch'))
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
