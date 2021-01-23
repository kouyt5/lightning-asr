from typing import List, Any, Dict
from comet_ml import Experiment  # 必须引入，不然报错
from exp_loggers import init_loggers
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning.metrics as metrics
import os
# from sklearn import metrics
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import get_mnist_pair
import logging
logger = logging.getLogger(__name__)


class LightingModule(pl.LightningModule):

    def forward(self, x):
        """
        用于推理阶段

        :param x: 传入参数 N*1*28*28
        :return: lists 预测的原始值
        """
        e = self.encoder(x)  # N*16*28*28
        e_view = e.view(e.size(0), -1)  # N*(16*28*28)
        d = self.decoder(e_view)  # N*10
        return d

    def configure_optimizers(self):
        """
        配置优化器

        :return: torch.optim
        """
        sgd_optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        adam_optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        lr_schulder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(adam_optim, T_0=20,
                                                                           T_mult=1, last_epoch=-1)
        return [adam_optim], [lr_schulder]

    def training_step(self, batch, batch_idx):
        """
        每一步的训练，相当于原生pytorch的

        >>> for i, batch in enumerate(DataLoader()):
        >>>     self.training_step(batch, i)

        :param batch: 包含每一batch的数据
        :param batch_idx: index
        :return: loss: torch.Tensor, 或者一个字典, 但必须包含'loss' key
        """
        x, y = batch
        d = self.forward(x)  # N*10
        loss = self.loss(d, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.acc(d, y),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        d = self.forward(x)  # N*10
        predict, true = get_mnist_pair(d, y)
        val_loss = self.loss(d, y)
        self.log('val_acc', self.acc(d, y),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss,
                 on_epoch=True, prog_bar=False, logger=True)
        return {
            'val_loss': val_loss,
            'images': batch[0],
            'val_acc': self.acc(d, y),
            'pred': predict,
            'true': true
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        验证一轮后调用，用于总结总的验证集效果

        :param outputs: 每一step的结果的列表
        :return: None
        """
        # 找出一轮中准确率最低的batch的前8个
        min_sample = min(outputs, key=lambda x: x['val_acc'])
        grid = torchvision.utils.make_grid(min_sample['images'][0:8], nrow=4)
        transform = torchvision.transforms.ToPILImage()
        image = transform(grid)
        pred = min_sample['pred'][0:8]
        true = min_sample['true'][0:8]
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
        logger.info(checkpoint.keys())

    def __init__(self, learning_rate=5e-3):
        super().__init__()
        self.acc = metrics.Accuracy()
        self.learning_rate = learning_rate
        self.save_hyperparameters('learning_rate')
        self.loss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28 * 16, 10),
            torch.nn.ReLU()
        )


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
    dataset = MNIST(data_cfg.get('path'), download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(data_cfg.get('path'), train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=128, num_workers=6)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=6)
    model = LightingModule()
    if tran_cfg.get('use_tpu'):  # 使用googleTPU训练
        trainer = pl.Trainer(tpu_cores=8,
                             logger=loggers,
                             callbacks=[checkpoint_callback, lr_callback],
                             resume_from_checkpoint=tran_cfg.get('checkpoint'),
                             # auto_lr_find=True,
                             max_epochs=tran_cfg.get('total_epoch'))
    trainer = pl.Trainer(gpus=tran_cfg.get('gpus'),
                         logger=loggers,
                         callbacks=[checkpoint_callback, lr_callback],
                         resume_from_checkpoint=tran_cfg.get('checkpoint'),
                         # auto_lr_find=True,
                         max_epochs=tran_cfg.get('total_epoch'))
    # trainer.tune(model, train_dataloader=train_loader)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
