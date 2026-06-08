"""
RelayFormer unified training entry point.

Supports pure-image, pure-video, and mixed image+video training through a single JSON
dataset config. The config is a flat list of dataset entries; image and video entries can
be freely combined:

    [
      ["ManiDataset",  "/path/to/CASIA2.0"],
      ["JsonDataset",  "/path/to/list.json"],
      ["VideoDataset", "/path/to/DAVIS",   ["OPN", "DVI"]]
    ]

  - only image entries  -> pure image training
  - only video entries  -> pure video training
  - both                -> mixed training
"""
import os
import json
import time
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import albumentations as albu
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter

import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.transforms import RandomCopyMove, RandomInpainting
from IMDLBenCo.evaluation import PixelF1, ImageF1  # noqa: F401  (ImageF1 available if needed)
from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.training_scripts.trainer import train_one_epoch

# Importing the package registers RelayFormer in the MODELS registry.
from models import RelayFormer  # noqa: F401
from datasets import build_dataset_from_config


def get_args_parser():
    parser = argparse.ArgumentParser('RelayFormer unified training', add_help=True)

    # Model
    parser.add_argument('--model', default='RelayFormer', type=str,
                        help='The name of applied model')
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Enable label prediction loss for models that support it.')

    # Dataset
    parser.add_argument('--image_size', default=1024, type=int,
                        help='Working resolution; the model input_size is forced to match this.')
    parser.add_argument('--if_padding', action='store_true', help='Pad all images to image_size.')
    parser.add_argument('--if_resizing', action='store_true', help='Resize all images to image_size.')
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (pixels) for the edge mask generator.')
    parser.add_argument('--clip_len', default=4, type=int, help='Frames per video clip.')
    parser.add_argument('--data_path', required=True, type=str,
                        help='Train dataset config JSON (flat list of dataset entries).')
    parser.add_argument('--test_data_path', default=None, type=str,
                        help='Test dataset config JSON: {name: dataset_entry, ...}. Optional.')

    # Training
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (samples = clips/images, not frames).')
    parser.add_argument('--test_batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--no_model_eval', action='store_true',
                        help='Do not use model.eval() during testing.')
    parser.add_argument('--test_period', default=4, type=int, help='Test every N epochs.')
    parser.add_argument('--save_period', default=25, type=int, help='Save checkpoint every N epochs.')
    parser.add_argument('--log_per_epoch_count', default=20, type=int)
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--if_not_amp', action='store_true', help='Disable automatic mixed precision.')
    parser.add_argument('--accum_iter', default=16, type=int, help='Gradient accumulation steps.')

    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base lr: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N')

    # Output / misc
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from a checkpoint path.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    args, remaining_args = parser.parse_known_args()

    # Build a secondary parser from the model's __init__ annotations.
    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args


def build_train_transform():
    return albu.Compose([
        albu.RandomScale(scale_limit=0.2, p=1),
        RandomCopyMove(p=0.1),
        RandomInpainting(p=0.1),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=1),
        albu.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
        albu.RandomRotate90(p=0.5),
        albu.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


def main(args, model_args):
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = build_train_transform()
    test_transform = albu.Compose([])  # add robustness perturbations here if desired
    print("Train transform:", train_transform)
    print("Test transform:", test_transform)

    # Optional post function registered as "<model>_post_func".
    post_function_name = f"{args.model.lower()}_post_func"
    try:
        post_function = POSTFUNCS.get_lower(post_function_name)
        print(f"Post function loaded: {post_function}")
    except Exception:
        print(f"Post function {post_function_name} not found, using None.")
        post_function = None

    common_ds_kwargs = dict(
        clip_len=args.clip_len,
        is_padding=args.if_padding,
        is_resizing=args.if_resizing,
        output_size=(args.image_size, args.image_size),
        edge_width=args.edge_mask_width,
        post_funcs=post_function,
    )

    # ---- Train dataset (image / video / mixed, auto-detected from config) ----
    dataset_train = build_dataset_from_config(
        args.data_path, split='train',
        image_transforms=train_transform,
        **common_ds_kwargs,
    )

    # ---- Test datasets: {name: dataset_entry} ----
    test_datasets = {}
    if args.test_data_path:
        with open(args.test_data_path, "r") as f:
            test_dataset_json = json.load(f)
        for dataset_name, dataset_entry in test_dataset_json.items():
            test_datasets[dataset_name] = build_dataset_from_config(
                [dataset_entry], split='test',
                image_transforms=test_transform,
                **common_ds_kwargs,
            )

    print(dataset_train)
    print(test_datasets)

    # ---- Samplers ----
    test_sampler = {}
    global_rank = misc.get_rank()
    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        for name, ds in test_datasets.items():
            test_sampler[name] = torch.utils.data.DistributedSampler(
                ds, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        for name, ds in test_datasets.items():
            test_sampler[name] = torch.utils.data.RandomSampler(ds)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True, collate_fn=dataset_train.collate_fn)

    test_dataloaders = {}
    for name in test_sampler.keys():
        test_dataloaders[name] = torch.utils.data.DataLoader(
            test_datasets[name], sampler=test_sampler[name],
            batch_size=args.test_batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True, collate_fn=test_datasets[name].collate_fn)

    # ---- Model (via registry) ----
    model_cls = MODELS.get(args.model)
    if isinstance(model_cls, (types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model_cls).parameters
    else:
        model_init_params = inspect.signature(model_cls.__init__).parameters
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    for k, v in vars(model_args).items():
        if k in model_init_params and k not in combined_args:
            combined_args[k] = v
    # Keep model resolution in sync with the dataset resolution.
    if 'input_size' in model_init_params:
        combined_args['input_size'] = args.image_size
    model = model_cls(**combined_args)

    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
        # ImageF1(threshold=0.5),
    ]

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module

    args.opt = 'AdamW'
    args.betas = (0.9, 0.999)
    args.momentum = 0.9
    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_evaluate_metric_value = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, log_per_epoch_count=args.log_per_epoch_count, args=args)

        if args.output_dir and (epoch % args.save_period == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        optimizer.zero_grad()

        # ---- Periodic evaluation ----
        if test_dataloaders and (epoch % args.test_period == 0 or epoch + 1 == args.epochs):
            values = {}
            for name, loader in test_dataloaders.items():
                print(f'!!!Start Test: {name}', len(loader))
                test_stats = test_one_epoch(
                    model, data_loader=loader, evaluator_list=evaluator_list,
                    device=device, epoch=epoch, name=name,
                    log_writer=log_writer, args=args, is_test=False)
                values[name] = {ev.name: test_stats[ev.name] for ev in evaluator_list}

            metrics_dict = {metric: {ds: values[ds][metric] for ds in values}
                            for metric in {m for d in values.values() for m in d}}
            metric_means = {metric: np.mean(list(ds.values())) for metric, ds in metrics_dict.items()}
            evaluate_metric_value = np.mean(list(metric_means.values()))

            if evaluate_metric_value > best_evaluate_metric_value:
                best_evaluate_metric_value = evaluate_metric_value
                print(f"Best {' '.join(ev.name for ev in evaluator_list)} = {best_evaluate_metric_value}")
                if args.output_dir and epoch > 20:
                    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            else:
                print(f"Average {' '.join(ev.name for ev in evaluator_list)} = {evaluate_metric_value}")

            if log_writer is not None:
                for metric, ds in metrics_dict.items():
                    log_writer.add_scalars(f'{metric}_Metric', ds, epoch)
                log_writer.add_scalar('Average', evaluate_metric_value, epoch)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
