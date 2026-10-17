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
    parser.add_argument(
        "--precision",
        choices= ["fp16", "bf16", "amp"],
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
    parser.add_argument(
        "--drop_path_rate",
        type= float,
        default=0.0,
        help="drop path rate."
    )

    ## train
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda"], 
        default="cuda"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--world_size", 
        type=int, 
        default=1
    )
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
        "--grad_clip_norm",
        type= float,
        default=None,
        help="Gradient clipping."
    )
    parser.add_argument(
        "--topk_rate",
        type= float,
        default=1.0,
        help="Only use top k% similarity protorpyes for training."
    )

    
    ## validation
    parser.add_argument(
        "--valid_batch_size",
        type=float,
        default=0.2,
        help="Validate batch size."
    )
    parser.add_argument(
        "--val_frequency",
        type=int,
        default=1,
        help="How often evaluate model per epochs with valid dataset."
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
        default=1e-04,
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
    parser.add_argument(
        "--patch_embed_lr_mult",
        type=float,
        default=1.0,
        help="Variable to multiply patch embed layer lr"
    )


    ## scheduler
    parser.add_argument(
        "--sch_name",
        choices= ["cos"],
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
        "--style_weight",
        type=list,
        default=[1.0, 1.0],
        help="weight of style loss. [Start, End] => The style weight moves from 'Start' to 'End' according to the cosine scheduler."
    )
    parser.add_argument(
        "--proto_weight",
        type=list,
        default=[1.0, 1.0],
        help="weight of prototype loss. [Start, End] => The prototype weight moves from 'Start' to 'End' according to the cosine scheduler."
    )
    parser.add_argument(
        "--proto_momentum",
        type=list,
        default=[0.9, 0.99],
        help="momentum for update the prototpyes vectors. [Start, End] => The prototpye momentum moves from 'Start' to 'End' according to the cosine scheduler."
    )
    parser.add_argument(
        "--BS_update_freq",
        type=int,
        default=1000,
        help="Batch sampler update frequency. Similarities between prototpyes are updated during training and used for hard class sampling in batch sampler."
    )
    parser.add_argument(
        "--BS_update_shape",
        type=float,
        default=1.0,
        help="Batch sampler update step shape. " \
        "shape < 1.0 => updates are occured more frequently in the first half, shape == 1.0 => Linear, " \
        "shape > 1.0 => updates are occured more frequently in the second half."
    )
    parser.add_argument(
        "--BS_update_start",
        type=float,
        default=0.1,
        help="When start first update of Batch sampler. Hard class sampling is start after first updates."
    )
    parser.add_argument(
        "--proto_ema_threshold_start_epoch",
        type=int,
        default=5,
        help="After the this epoch, Only samples that similarity exceed the threshold are used for prototpye ema update."
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
        help="Validation data that the type of dataframe with a .parquet extension."
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test data that the type of dataframe with a .parquet extension."
    )
    parser.add_argument(
        "--relation",
        type=str,
        default=None,
        help="Relation labels between classes. "
    )
    parser.add_argument(
        "--feat_label",
        type=str,
        default=None,
        help="style features labels. "
    )
    parser.add_argument(
        "--feat_class_weight_path",
        type=str,
        default=None,
        help="style features labels class weight. "
    )
    parser.add_argument(
        "--feat_pos_weight_path",
        type=str,
        default=None,
        help="style features labels pos weight. "
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader num_wokers."
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

    # sampler
    parser.add_argument(
        "--hard_num",
        type=list,
        default=[1, 2],
        help="How many hard class samples include in one batch."
    )
    parser.add_argument(
        "--same_num",
        type=list,
        default=[1, 2],
        help="How many same class samples include in one batch."
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