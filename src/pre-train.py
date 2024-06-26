import hydra
import lightning.pytorch as pl
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import Trainer


@hydra.main(version_base="1.3", config_path="../configs", config_name="pre-train.yaml")
def main(cfg: DictConfig):
    if cfg.seed:
        pl.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, model=model)
    trainer.fit(datamodule)
    print(trainer.test(datamodule))


if __name__ == "__main__":
    main()
