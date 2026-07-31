import argparse

def setup_parser():
    parser = argparse.ArgumentParser()

    ## model
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the vision backbone to use.",
    )
    # parser.add_argument(
    #     "--precision",
    #     choices= ["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
    #     default='amp',
    #     help="Floating point precision."
    # )
    parser.add_argument(
        "--precision",
        choices= ["fp16", "bf16"],
        default=None,
        help="Floating point precision."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to latest checkpoint that want to resume train."
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="path to pretrained weight."
    )
    parser.add_argument(
        "--save_frequency",
        type= int,
        default=1,
        help="How often to save checkpoints."
    )
    parser.add_argument(
        "--max_save_count",
        type= int,
        default=None,
        help="How many checkpoints to save. Checkpoint will be deleted starting from the oldest one."
    )

    ## train
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU."
    )
    parser.add_argument(
        "--accum_freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        help="Model drop_path rate."
    )
    parser.add_argument(
        "--class_weight",
        type=float,
        default=1.0,
        help="weight value argument of BCEWithLogitsLoss."
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="pos_weight value argument of BCEWithLogitsLoss. eq: number of 0 labels / number of 1 labels * pos_weight"
    )
    parser.add_argument(
        "--bce_weight",
        type=float,
        default=1.0,
        help="loss weight of BCE loss."
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=1.0,
        help="loss weight of MSE loss."
    )
    parser.add_argument(
        "--mcw_by_score",
        type= float,
        default=1.0,
        help="the weight to set higher loss for higher label. equation: loss = e^{mse_class_weight * label} * mse_loss"
    )
    parser.add_argument(
        "--count_weight",
        type= bool,
        default=False,
        help="focus lower frequencies samples for train."
    )
    parser.add_argument(
        "--cw_bins",
        type= float,
        default=10,
        help="how many intervals to divide the label range. when count_weight is true."
    )
    parser.add_argument(
        "--cw_alpha",
        type= float,
        default=0.5,
        help="higher value means more focus on lower frequencies samples. when count_weight is true."
    )
    parser.add_argument(
        "--grad_clip_norm",
        type= float,
        default=None,
        help="Gradient clip."
    )

    ## loss
    parser.add_argument(
        "--same_class_weight",
        type= float,
        default=1.0,
        help="the label to set positive pair except for self."
    )
    parser.add_argument(
        "--recon_weight",
        type= float,
        default=1.0,
        help="Reconstruction loss weight."
    )


    ## validation
    parser.add_argument(
        "--valid_batch_size",
        type=float,
        default=0.2,
        help="Validate batch size."
    )
    parser.add_argument(
        "--valid_size",
        type=int,
        default=2000,
        help="Validation set size."
    )
    parser.add_argument(
        "--val_frequency",
        type=int,
        default=1,
        help="How often evaluate model per epochs with valid dataset."
    )

     ## logger
    parser.add_argument(
        "--report_to",
        type=str,
        default='tensorboard',
        help="Options are ['wandb', 'tensorboard']."
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default='my-clip',
        help="wandb project name."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="wandb run id. Must set this argument when use resume argument."
    )

    # optimizer
    parser.add_argument(
        "--opt_name",
        choices=["adamw", "adamw8bit", "dadaptation", "prodigy"],
        default="adamw8bit",
        help="optimizer name. ex: adamw, adamw8bit, dadaptation, prodigy"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.05,
        help="Weight decay."
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.09,
        help="Adam beta 1."
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta 2."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Adam epsilon."
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=1.0,
        help="Layer-wise learning rate decay."
    )

    ## scheduler
    parser.add_argument(
        "--sch_name",
        choices= ["cos", "cos_restart"],
        default='cos',
        help="Type of leanring rate scheduler."
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default='1e-8',
        help="minimum learning rate for scheduler. If not set, it is set to 1e-8."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Warmup ratio in total steps. when use restart scheduler, it is calculated as (warmup = warmup_ratio * total_steps / cycles)."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Restart count for restart scheduler."
    )
    parser.add_argument(
        "--cycle_multiplier",
        type=float,
        default=1.0,
        help="Each cycle step count is multiplied by this value after first cycle."
    )
    parser.add_argument(
        "--re_g", # restart gamma
        type=float,
        default=1.0,
        help="maximum learning rate is decreased by this value after first cycle."
    )
    

    ## data
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Training data that the type of dataframe with a .parquet extension."
    )
    parser.add_argument(
        "--valid",
        type=str,
        default=None,
        help=
            "Validation data that the type of dataframe with a .parquet extension."
    )
    parser.add_argument(
        "--relation",
        type=str,
        default=None,
        help="Relation labels between classes. "
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="image",
        help="For pd.dataframe, the name of column contains image path."
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="annotation",
        help="For pd.dataframe, the name of column contains label(str)."
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="label",
        help=(
            "For pd.dataframe, the name of column contains label(numpy.ndarray). "
            "ex: [0, 1, 1, 0]"
        )
    )
    parser.add_argument(
        "--mse_target_key",
        type=str,
        default="mse_label",
        help=(
            "For pd.dataframe, the name of column contains mse label(numpy.ndarray). "
            "ex: [0, 1, 2, 0]"
        )
    )
    parser.add_argument(
        "--label_json",
        type=str,
        default=None,
        help="List the types of classes."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader num_wokers."
    )
    parser.add_argument(
        "--mos_range",
        type= list,
        default=None,
        help="the range of quality score."
    )
    parser.add_argument(
        "--balanced_sample",
        type= bool,
        default=False,
        help="Resample to balance the class."
    )

    ## dataset
    parser.add_argument(
        "--image_size",
        type=list,
        default=[448, 448],
        help="Image size for train."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Model patch size for make frame."
    )
    parser.add_argument(
        "--train_aug",
        type=str,
        default="long_max_pad_train",
        help="Train dataset transforms."
    )
    parser.add_argument(
        "--valid_aug",
        type=str,
        default="long_max_pad_valid",
        help="Valid dataset transforms."
    )
    parser.add_argument(
        "--prototype_path",
        type=str,
        default=None,
        help="Each class prototopye vectors path for initialize."
    )

    ## debug
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Debug option."
    )

    ## Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Load arguments from config file."
    )

    return parser