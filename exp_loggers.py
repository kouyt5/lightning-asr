from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
import hydra
from omegaconf import DictConfig, OmegaConf


__all__ = ['init_loggers']


def init_loggers(cfg: DictConfig):
    comet_cfg = cfg.get('comet')
    tensorboard_cfg = cfg.get('tensorboard')
    comet_logger = CometLogger(
        api_key=comet_cfg.get('COMET_API_KEY'),
        workspace=comet_cfg.get('workspace'),
        project_name=comet_cfg.get('project_name'),  # Optional
        experiment_name=comet_cfg.get('experiment_prefix_name')+comet_cfg.get('experiment_fixed_name'),  # Optional
        experiment_key=comet_cfg.get('experiment_key') # restore previous experiment
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=tensorboard_cfg.get("save_dir"),
        name=tensorboard_cfg.get("name")
    )
    comet_logger.experiment.log_code(file_name=None, folder='../../../../')
    return tensorboard_logger, comet_logger


@hydra.main(config_path='conf', config_name='conf')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg=cfg))
    logger_cfg = cfg.get('loggers')
    loggers = init_loggers(cfg=logger_cfg)


if __name__ == '__main__':
    main()
