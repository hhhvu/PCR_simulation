import argparse
import sys
import os
import shutil 
from os.path import dirname, realpath
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.append(dirname(dirname(realpath(__file__))))
from src.pcr_lightning import FusionModel, GeneFusionModel, GeneFusionHeadsModel, GeneEnsembleModel, CurveShapeModel, CurveShapeDeltaModel, SeqModel, SeqDeltaModel, SeqCurveModel, SeqDeltaGeneModel, SeqGeneModel, SeqCurveGeneModel
from src.pcr_dataset import ImageSequenceDataModule, ImageSequenceGeneDataModule, ImageDataModule, SequenceDataModule
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.accelerators import find_usable_cuda_devices
import lightning.pytorch as pl

NAME_TO_MODEL_CLASS = {
    "fusion": FusionModel,
    "gene_fusion": GeneFusionModel,
    "gene_fusion_heads": GeneFusionHeadsModel,
    "gene_ensemble": GeneEnsembleModel,
    "curve": CurveShapeModel,
    "curve_delta": CurveShapeDeltaModel,
    "seq": SeqModel,
    "seq_delta": SeqDeltaModel,
    "seq_curve": SeqCurveModel,
    "seq_delta_gene": SeqDeltaGeneModel,
    "seq_gene": SeqGeneModel,
    "seq_curve_gene": SeqCurveGeneModel
}

NAME_TO_DATASET_CLASS = {
    "imgseq": ImageSequenceDataModule,
    "imgseqgene": ImageSequenceGeneDataModule,
    "img": ImageDataModule,
    'seq_data': SequenceDataModule
}

# CUDA_VISIBLE_DEVICES=0 python scripts/pcr_train_script.py --project_name pcr-classification --experiment_name FusionModel_Test --train --trainer.max_epochs 10
#python scripts/pcr_train_script.py --project_name pcr-classification_fusion_test --experiment_name FusionModel_Test --train --trainer.max_epochs 10 --fusion.pretrained true --fusion.init_lr 1e-5 --fusion.hidden_size 512 --fusion.latent_dim 512 --fusion.num_layers 10  --model_name fusion

#python scripts/pcr_train_script.py --project_name pcr-classification_fusion_test --experiment_name FusionModel_Test --train --trainer.max_epochs 1 --gene_fusion.init_lr 1e-5 --gene_fusion.hidden_size 512 --gene_fusion.latent_dim 32 --gene_fusion.num_layers 10  --model_name gene_fusion --dataset_name imgseqgene --grid_search

#python scripts/pcr_train_script.py --project_name pcr-classification_Seq_large --experiment_name SeqModelLargeData_Test --train --trainer.max_epochs 50 --model_name seq --dataset_name seq_data --grid_search --igi_call true
#python scripts/pcr_train_script.py --project_name pcr-classification_Seq_large --experiment_name SeqModelOldData_Test --train --trainer.max_epochs 50 --model_name seq --dataset_name seq_data --grid_search --igi_call true

#conda create -n PCR_v2 python=3.9

# Train command
# CUDA_VISIBLE_DEVICES=7 python scripts/pcr_train_script.py --project_name pcr-classification_fusion_test --experiment_name GeneFusionHeadsModel_Test --train --trainer.max_epochs 50 --gene_fusion_heads.init_lr 1e-4 --gene_fusion_heads.hidden_size 512 --gene_fusion_heads.latent_dim 512 --gene_fusion_heads.num_layers 3 --gene_fusion_heads.delta 64 --model_name gene_fusion_heads --dataset_name imgseqgene --igi_call true
# python -m pdb -c continue scripts/pcr_train_script.py --train --grid_search --project_name pcr-classification_image --experiment_name ImgModel_Test --trainer.max_epochs 50 --model_name curve --dataset_name img --igi_call true
# CUDA_VISIBLE_DEVICES=1 python -m pdb -c continue scripts/pcr_train_script.py --train --grid_search --project_name pcr-classification_image_range --experiment_name ImgRangeModel --trainer.max_epochs 50 --model_name curve_delta --dataset_name imgseq --igi_call true
# python -m pdb -c continue scripts/pcr_train_script.py --project_name pcr-classification_image --experiment_name ImgModel_Save_3_head --trainer.max_epochs 100 --model_name curve --dataset_name img --igi_call true --curve.latent_dim 1024 --curve.init_lr 1e-5 --checkpoint_path pcr-classification_image/d73jmc6v/checkpoints/epoch=11-step=660.ckpt 
# python -m pdb -c continue scripts/pcr_train_script.py --train --project_name pcr-classification_image --experiment_name ImgModel_Save_3_head --trainer.max_epochs 100 --model_name curve --dataset_name img --igi_call true --curve.latent_dim 1024 --curve.init_lr 1e-5

# Eval command
# CUDA_VISIBLE_DEVICES=7 python scripts/pcr_train_script.py --project_name pcr-classification_fusion_test --experiment_name GeneFusionHeadsModel_Test --checkpoint_path pcr-classification_fusion_test/kibnmcmb/checkpoints/epoch=43-step=19184.ckpt --gene_fusion_heads.init_lr 1e-4 --gene_fusion_heads.hidden_size 512 --gene_fusion_heads.latent_dim 512 --gene_fusion_heads.num_layers 3 --gene_fusion_heads.delta 64 --model_name gene_fusion_heads --dataset_name imgseqgene --igi_call true

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="fusion",
        help="Name of model to use.",
    )

    parser.add_argument(
        "--dataset_name",
        default="imgseqgene",
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
        default="val_auc",
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

    parser.add_argument(
        "--grid_search",
        default=False,
        action="store_true",
        help="Whether to save model checkpoints. No saving during grid search."
    )

    parser.add_argument(
        "--igi_call",
        default='True',
        help="Whether to include igi_call information (multiple heads of output)."
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
    # dataset_args['use_data_augmentation'] = bool(args.use_data_augmentation)
    dataset_args['batch_size'] = int(args.batch_size)
    dataset_args['curve_dict_path'] =  'data/new_groundtruth_df_curve_dict_fn.pkl' #'data/groundtruth_df_curve_dict_split_v2.pkl' 
    dataset_args['target_df_path'] = 'data/new_groundtruth_df_target_data.csv' #'data/groundtruth_df_target_data_split_v2.csv
    dataset_args['igi_call'] = (args.igi_call == 'true')

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
                                    entity="saselvan",
                                    log_model="all")

    args.trainer.accelerator = 'auto'
    args.trainer.logger = logger
    # args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended

    if args.grid_search:
        args.trainer.enable_checkpointing=False
    else:
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

    #print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)
    val_results = model.validation_outputs
    #print("******val results******", val_results)

    # print("Evaluating model on test set")
    # trainer.test(model, datamodule)
    

    # test_results = trainer.results
    # print("******test results******", test_results)

    #if args.grid_search:
        # PyTorch Lightning creates checkpoint folder
        #ckpt_dir = os.path.dirname(trainer.checkpoint_callback.best_model_path) 

        # Delete entire checkpoint folder
        # shutil.rmtree(ckpt_dir)

    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
