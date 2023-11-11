
import argparse
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from src.pcr_lightning import FusionModel, GeneFusionModel, GeneEnsembleModel
from src.pcr_dataset import ImageSequenceDataModule, ImageSequenceGeneDataModule
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl

NAME_TO_MODEL_CLASS = {
    "fusion": FusionModel,
    "gene_fusion": GeneFusionModel,
    "gene_ensemble": GeneEnsembleModel,
}

NAME_TO_DATASET_CLASS = {
    "imgseq": ImageSequenceDataModule,
    "imgseqgene": ImageSequenceGeneDataModule,
}

# TO DELETE
# tmux new-session -s cph_200a
# conda create -n cph_200a python=3.10
# python scripts/main.py --project_name cornerstone_mlp --train --trainer.max_epochs 100
# python scripts/main.py --project_name cornerstone_cnn --train --trainer.max_epochs 100 --cnn.use_bn True --use_data_augmentation True --batch_size 256 --model_name "cnn" --project_name "cornerstone-cnn" --cnn.num_layers 4

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="fusion",
        help="Name of model to use.",
    )

    parser.add_argument(
        "--dataset_name",
        default="imgseq",
        help="Name of dataset to use."
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        help="Which batch size to use during training"
    )

    parser.add_argument(
        "--project_name",
        default=None,
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Name of experiment for wandb"
    )

    parser.add_argument(
        "--monitor_key",
        default="val_loss",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="Whether to train the model."
    )

    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    for model_name, model_class in NAME_TO_MODEL_CLASS.items():
        parser.add_lightning_class_args(model_class, nested_key=model_name)
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    print(args)
    print("Loading data ..")

    print("Preparing lightning data module (encapsulates dataset init and data loaders)")
    """
        Most the data loading logic is pre-implemented in the LightningDataModule class for you.
        However, you may want to alter this code for special localization logic or to suit your risk
        model implementations
    """
    #TODO think of a cleaner way to do this
    # args.dataset_name.use_data_augmentation = args.use_data_augmentation
    # args.dataset_name.batch_size = args.batch_size

    dataset_args = vars(args[args.dataset_name])
    dataset_args['use_data_augmentation'] = bool(args.use_data_augmentation)
    dataset_args['batch_size'] = int(args.batch_size)
    dataset_args['curve_dict_path'] = 'data/groundtruth_df_curve_dict_split_v2.pkl'
    dataset_args['target_df_path'] = 'data/groundtruth_df_target_data_split_v2.csv'

    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**dataset_args)
    # datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**vars(args[args.dataset_name]))

    print("Initializing model")
    ## TODO: Implement your deep learning methods
    if args.checkpoint_path is None:
        model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
    else:
        model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    print("Initializing trainer")
    logger = pl.loggers.WandbLogger(project=args.project_name, 
                                    name = args.experiment_name,
                                    entity="test_entity")

    args.trainer.accelerator = 'auto'
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            save_last=True
        )]

    trainer = pl.Trainer(**vars(args.trainer))
    trainer.log_every_n_steps = 1

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)

    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
