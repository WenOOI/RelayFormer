"""
RelayFormer unified testing entry point.

Evaluates a checkpoint across image and/or video test sets described by a JSON config:

    {
      "CASIA1.0": ["ManiDataset",  "/path/to/CASIA1.0"],
      "coverage": ["JsonDataset",  "/path/to/coverage.json"],
      "MOSE":     ["VideoDataset", "/path/to/MOSE", ["E2FGVI", "FuseFormer", "STTN"]]
    }
"""
import os
import json
import time
import types
import inspect
import argparse
import datetime
from pathlib import Path

import torch
import albumentations as albu
from torch.utils.tensorboard import SummaryWriter

import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.evaluation import PixelF1, ImageF1  # noqa: F401
from IMDLBenCo.training_scripts.tester import test_one_epoch

from models import RelayFormer  # noqa: F401  (registers the model)
from datasets import build_dataset_from_config


def get_args_parser():
    parser = argparse.ArgumentParser('RelayFormer unified testing', add_help=True)
    parser.add_argument('--model', default='RelayFormer', type=str)
    parser.add_argument('--if_predict_label', action='store_true')

    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--if_padding', action='store_true')
    parser.add_argument('--if_resizing', action='store_true')
    parser.add_argument('--edge_mask_width', default=None, type=int)
    parser.add_argument('--clip_len', default=4, type=int)
    parser.add_argument('--test_data_json', required=True, type=str,
                        help='Test dataset config JSON: {name: dataset_entry, ...}.')

    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Path to a .pth checkpoint to evaluate.')
    parser.add_argument('--test_batch_size', default=2, type=int)
    parser.add_argument('--no_model_eval', action='store_true')
    parser.add_argument('--merge_lora', action='store_true',
                        help='Merge LoRA weights before evaluation for faster inference.')

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    args, remaining_args = parser.parse_known_args()
    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)
    return args, model_args


def main(args, model_args):
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)

    test_transform = albu.Compose([])

    with open(args.test_data_json, "r") as f:
        test_dataset_json = json.load(f)

    global_rank = misc.get_rank() if args.distributed else 0

    # ---- Model ----
    model_cls = MODELS.get(args.model)
    if isinstance(model_cls, (types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model_cls).parameters
    else:
        model_init_params = inspect.signature(model_cls.__init__).parameters
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    for k, v in vars(model_args).items():
        if k in model_init_params and k not in combined_args:
            combined_args[k] = v
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
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # ---- Load checkpoint ----
    print("Loading checkpoint: %s" % args.checkpoint_path)
    ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    model_without_ddp.load_state_dict(ckpt['model'])
    if args.merge_lora:
        model_without_ddp.merge_lora()
        print("Merged LoRA weights for inference.")

    post_function_name = f"{args.model.lower()}_post_func"
    try:
        post_function = POSTFUNCS.get_lower(post_function_name)
    except Exception:
        post_function = None

    common_ds_kwargs = dict(
        clip_len=args.clip_len,
        is_padding=args.if_padding,
        is_resizing=args.if_resizing,
        output_size=(args.image_size, args.image_size),
        edge_width=args.edge_mask_width,
        post_funcs=post_function,
    )

    start_time = time.time()
    for dataset_name, dataset_entry in test_dataset_json.items():
        full_log_dir = os.path.join(args.log_dir, dataset_name)
        if global_rank == 0:
            os.makedirs(full_log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=full_log_dir)
        else:
            log_writer = None

        dataset_test = build_dataset_from_config(
            [dataset_entry], split='test', image_transforms=test_transform, **common_ds_kwargs)
        print(dataset_test)
        print("len(dataset_test)", len(dataset_test))

        if args.distributed:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=misc.get_world_size(), rank=global_rank,
                shuffle=False, drop_last=True)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        dataloader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test, batch_size=args.test_batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
            collate_fn=dataset_test.collate_fn)

        print("Testing on dataset: %s" % dataset_name)
        test_stats = test_one_epoch(
            model=model, data_loader=dataloader_test, evaluator_list=evaluator_list,
            device=device, epoch=0, name=dataset_name, log_writer=log_writer, args=args)

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'dataset': dataset_name}
        if global_rank == 0:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        print(f"[{dataset_name}] {log_stats}")

    total_time = time.time() - start_time
    print('Total testing time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
