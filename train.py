from typing import Callable, List, Any, Dict, Optional
from comet_ml import Experiment
from torch.optim.optimizer import Optimizer  # 必须引入，不然报错
from exp_loggers import init_loggers, get_comet_experiment
import torch
import pytorch_lightning as pl
from data_module import LibriDataModule
from scheduler.cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
from scheduler.lr_policy import CosineAnnealing
import os
import hydra
from omegaconf import DictConfig, OmegaConf
#from models.QuartNet import MyModel2
from models.QuartNetContext import MyModel2
from utils.asr_metrics import WER
import logging
from scheduler.novograd import Novograd
# import torch_ort

logger = logging.getLogger(__name__)


class LightingModule(pl.LightningModule):

    def forward(self, inputs, percentage):
        """
        用于推理阶段

        :param 
            inputs: 传入参数 N*1*64×L
            percent: 比例
        :return: lists 预测的原始值
        """
        return self.encoder(inputs, percentage)  # N*L'*C

    def configure_optimizers(self):
        """
        配置优化器

        :return: torch.optim
        """
        self.print('设置学习率' + str(self.learning_rate))
        # sgd_optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # adam_optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        novo_optim = Novograd(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay, betas=(0.8, 0.5))
        # self.cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(novo_optim, T_0=5,
        #                                                                    T_mult=2, last_epoch=-1)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(novo_optim, mode='min',
        #                                                          factor=0.1, patience=10,
        #                                                          threshold=1e-4, threshold_mode='rel',
        #                                                          cooldown=3, min_lr=1e-4)
        lr_scheduler = CosineAnnealingWarmupRestarts(novo_optim, first_cycle_steps=self.total_epoch*len(self.train_dataloader()),
                                                   cycle_mult=2, max_lr=self.learning_rate, min_lr=1e-4,
                                                   warmup_steps=1000, gamma=0.5)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(novo_optim, gamma=0.98)
        pack_scheduer = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
        }
        return [novo_optim], [pack_scheduer]

    def training_step(self, batch, batch_idx):
        """
        每一步的训练，相当于原生pytorch的
        :param batch: 包含每一batch的数据
        :param batch_idx: index
        :return: loss: torch.Tensor, 或者一个字典, 但必须包含'loss' key
        """
        input = batch[0]
        percentage = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = self.encoder(input, percentage)
        t_lengths = torch.mul(out.size(1), percentage).int()  # 输出实际长度
        loss = torch.mean(self.loss(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_wer', self.wer(out.argmax(dim=-1, keepdim=False), trans, trans_lengths, t_lengths),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % 50 == 0:
            print('\n')
            logging.info("pred:"+self.wer.ctc_decoder_predictions_tensor(torch.argmax(out, dim=-1, keepdim=False), t_lengths)[0])
            logging.info("true:"+self.wer.decode_reference(trans, trans_lengths)[0])
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch[0]
        percentage = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = self.encoder(input, percentage)
        t_lengths = torch.mul(out.size(1), percentage).int()  # 输出实际长度
        loss = torch.mean(self.loss(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths))
        wer = self.wer(out.argmax(dim=-1, keepdim=False), trans, trans_lengths, t_lengths)
        self.log('val_wer', wer,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return_val = {
            'val_loss': loss,
            'input': batch[0],
            'val_wer': wer,
            'pred': self.wer.ctc_decoder_predictions_tensor(torch.argmax(out, dim=-1, keepdim=False), t_lengths),
            'true': self.wer.decode_reference(trans, trans_lengths),
            'path': batch[-1]
        }
        if batch_idx % 50 == 0:
            print('\n')
            logging.info("pred:"+self.wer.ctc_decoder_predictions_tensor(torch.argmax(out, dim=-1, keepdim=False), t_lengths)[0])
            logging.info("true:"+self.wer.decode_reference(trans, trans_lengths)[0])
            print('\n')
            logging.info("pred:"+self.wer.ctc_decoder_predictions_tensor(torch.argmax(out, dim=-1, keepdim=False), t_lengths)[-1])
            logging.info("true:"+self.wer.decode_reference(trans, trans_lengths)[-1])
        return return_val

    def test_step(self, batch, batch_idx):
        input = batch[0]
        percentage = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = self.encoder(input, percentage)
        t_lengths = torch.mul(out.size(1), percentage).int()  # 输出实际长度
        loss = torch.mean(self.loss(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths))
        return_val = {
            'test_loss': loss,
            'input': batch[0],
            'test_wer': self.wer(out.argmax(dim=-1, keepdim=False), trans, trans_lengths, t_lengths),
            'pred': self.wer.ctc_decoder_predictions_tensor(torch.argmax(out, dim=-1, keepdim=False), t_lengths),
            'true': self.wer.decode_reference(trans, trans_lengths),
            'path': batch[-1]
        }
        return return_val

    def test_epoch_end(
        self, outputs: List[Any]
    ) -> None:
        count = 0
        total_wer = 0.
        for item in outputs:
            total_wer += item['test_wer']
            count += 1
        logger.info('测试wer：'+str(total_wer/(count+1e-9)))

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        验证一轮后调用，用于总结总的验证集效果

        :param outputs: 每一step的结果的列表
        :return: None
        """
        # 找出一轮中准确率最低的batch的前8个
        # min_sample = min(outputs, key=lambda x: x['val_acc'])
        # grid = torchvision.utils.make_grid(min_sample['images'][0:8], nrow=4)
        # transform = torchvision.transforms.ToPILImage()
        # image = transform(grid)
        # pred = min_sample['pred'][0:8]
        # true = min_sample['true'][0:8]
        # 使用comet的logger
        # self.logger[1].experiment.log_image(image_data=image, name=str(pred)+str(true)+'.png')
        count = 0
        total_wer = 0.
        for item in outputs:
            total_wer += item['val_wer']
            count += 1
        logger.info('验证集wer：'+str(total_wer/(count+1e-9)))

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
        logger.info("保存一个checkpoint epoch={:d}".format(self.current_epoch))

    def __init__(self, learning_rate=5e-3, weight_decay=1e-4, labels=None,
                 total_epoch=50, drop_rate: float = 0., mask: bool = False,
                 use_cer=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.total_epoch = total_epoch
        self.save_hyperparameters()
        self.wer = WER(vocabulary=self.labels, use_cer=use_cer)
        self.loss = torch.nn.CTCLoss(blank=len(self.labels), reduction='none')  # 最后一个作为black
        self.encoder = MyModel2(labels=self.labels, drop_rate=drop_rate, mask=mask)
        # self.encoder = torch_ort.ORTModule(MyModel2(labels=self.labels, drop_rate=drop_rate, mask=mask))


@hydra.main(config_path='conf', config_name='conf')
def main(cfg: DictConfig):
    pl.seed_everything(0)  # 设置随机数seed
    print(OmegaConf.to_yaml(cfg=cfg))
    print(os.getcwd())
    tran_cfg = cfg.get('train')
    logger_cfg = cfg.get('loggers')
    data_cfg = cfg.get('data')
    model_cfg = cfg.get('model')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", monitor='val_wer', verbose=True,
                                                       save_last=True, save_top_k=3,
                                                       filename="asr-{epoch:02d}-{val_wer:.2f}")
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    loggers = init_loggers(cfg=logger_cfg)
    labels = data_cfg.get('labels')
    use_cer = False
    if isinstance(data_cfg.get('labels'), str):
        labels = [c.strip() for c in open(data_cfg.get('labels'), 'r').readlines()]
        use_cer = True
    data_module = LibriDataModule(data_cfg.get('train_manifest'), data_cfg.get('val_manifest'),
                                  labels=labels, train_bs=tran_cfg.get('train_batch_size'),
                                  dev_bs=tran_cfg.get('dev_batch_size'), test_manifest=data_cfg.get('test_manifest'),
                                  num_worker=data_cfg.get('num_worker'), 
                                  train_max_duration=data_cfg.get('train_max_duration'),
                                  dev_max_duration=data_cfg.get('dev_max_duration'))
    model = LightingModule(learning_rate=tran_cfg.get("learning_rate"),
                           weight_decay=tran_cfg.get("weight_decay"),
                           labels=labels,
                           total_epoch=tran_cfg.get('total_epoch'),
                           drop_rate=model_cfg.get('drop_rate'),
                           mask=model_cfg.get('mask'),
                           use_cer=use_cer)
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
                            limit_val_batches=1.0,
                            limit_train_batches=1.0,
                            max_epochs=tran_cfg.get('total_epoch'),
                            check_val_every_n_epoch=tran_cfg.get('check_val_every_n_epoch', 1),
                            gradient_clip_val=0,
                            gradient_clip_algorithm='value',
                            num_nodes=tran_cfg.get('num_nodes'))
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, test_dataloaders=data_module.test_dataloader())


if __name__ == '__main__':
    main()
