import hydra
import lightning.pytorch as pl
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="prune.yaml")
def main(cfg: DictConfig):
    if cfg.seed:
        pl.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, model=model)

    if cfg.ckpt_path:
        model = trainer.load(cfg.ckpt_path)

    amount = cfg.prune_amount
    total_steps = cfg.prune_steps
    amount_per_step = 1 - (1 - amount) ** (1 / total_steps)

    for _ in range(total_steps):
        trainer.prune(amount=amount_per_step, type=cfg.prune_type)
        trainer.fit(datamodule)
        print(trainer.test(datamodule))


if __name__ == "__main__":
    main()
