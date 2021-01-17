from typing import List, Any, Dict

from comet_ml import Experiment
from exp_loggers import init_loggers
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
from sklearn import metrics
import hydra
from omegaconf import DictConfig, OmegaConf


def get_pair(pred, true) -> (list, list):
    """
    获取两个预测标签和真实标签对应的list

    :param pred: 预测值
    :param true: 真是值
    :return: [pre_list, true_list]
    """
    y = torch.argmax(pred, dim=-1)
    y_list = y.detach().cpu().numpy().tolist()
    true_list = true.detach().cpu().numpy().tolist()
    return y_list, true_list


class LightingModle(pl.LightningModule):

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
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)

    def training_step(self, batch, batch_idx):
        """
        每一步的训练，相当于原生pytorch的

        >>> for i, batch in enumerate(DataLoader()):
        >>>     self.training_step(batch, i)

        :param batch: 包含每一batch的数据
        :param batch_idx: index
        :return: torch.Tensor, 这表示loss, 或者一个字典, 但必须包含'loss' key
        """
        x, y = batch
        d = self.forward(x)  # N*10
        loss = self.loss(d, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', metrics.accuracy_score(*get_pair(d, y)),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for param in self.optimizers().param_groups:
            self.log('lr', param['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        d = self.forward(x)  # N*10
        predit, true = get_pair(d, y)
        val_loss = self.loss(d, y)
        self.log('val_acc', metrics.accuracy_score(predit, true),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss,
                 on_epoch=True, prog_bar=False, logger=True)
        return {
            'val_loss': val_loss,
            'images': batch[0],
            'val_acc': metrics.accuracy_score(predit, true),
            'pred': predit,
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
        保存checkpoint后调用

        :param checkpoint: 状态字典

        .. code-block:: python
            字典keys = ['epoch', 'global_step', 'pytorch-lightning_version',
            'callbacks', 'optimizer_states', 'lr_schedulers',
            'state_dict', 'hparams_name', 'hyper_parameters']

        :return: None
        """
        print(checkpoint.keys())

    def __init__(self, learning_rate=5e-3):
        super().__init__()
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
            torch.nn.Linear(28*28*16, 10),
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
    checkpoint_callpoint = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", monitor='val_loss',verbose=True,
                                                        save_last=True, save_top_k=3,
                                                        filename="minst-{epoch:02d}-{val_loss:.2f}")
    loggers = init_loggers(cfg=logger_cfg)
    dataset = MNIST(data_cfg.get('path'), download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(data_cfg.get('path'), train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=128, num_workers=6)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=6)
    model = LightingModle()
    trainer = pl.Trainer(gpus=1,
                         logger=loggers,
                         callbacks=[checkpoint_callpoint],
                         resume_from_checkpoint=tran_cfg.get('checkpoint'),
                         # auto_lr_find=True,
                         max_epochs=tran_cfg.get('total_epoch'))
    # trainer.tune(model, train_dataloader=train_loader)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()

