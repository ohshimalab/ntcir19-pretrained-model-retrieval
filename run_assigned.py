#!/usr/bin/env python3
"""Assign dataset/model jobs to a machine using simple modulo slicing.

Usage example:
  python run_assigned.py --machine-id 0 --num-machines 5 --config config.toml --dry-run

This script enumerates the Cartesian product of data directories under
`[finetune].data_dir_root` and models listed in `[finetune].model_list_excel`.
It computes a linear job index as `dataset_idx * num_models + model_idx` and
assigns a job to a machine when `linear_index % num_machines == machine_id`.

If `--dry-run` is provided, the script only prints assigned pairs.
Otherwise it invokes `trainer.run_experiment` for each assigned pair.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ntcir19_pretrained_model_retrieval.config import load_config
from ntcir19_pretrained_model_retrieval.logger_setup import get_logger
from ntcir19_pretrained_model_retrieval.trainer import run_experiment


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run assigned dataset/model pairs using modulo slicing")
    p.add_argument("--machine-id", type=int, required=True, help="Machine id (0..num_machines-1)")
    p.add_argument("--num-machines", type=int, required=True, help="Total number of machines")
    p.add_argument("--config", type=Path, default=Path("config.toml"), help="Path to config file")
    p.add_argument("--dry-run", action="store_true", help="Only print assigned pairs, do not run experiments")
    p.add_argument("--log-file", type=Path, default=Path("finetune_assigned.log"), help="Path to log file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger(log_file=args.log_file)

    cfg = load_config(args.config)
    ft = cfg.finetune

    if ft.data_dir_root is None or ft.model_list_excel is None:
        logger.error("Config [finetune] must set data_dir_root and model_list_excel")
        sys.exit(2)

    data_dirs = [p for p in ft.data_dir_root.iterdir() if p.is_dir()]
    data_dirs = sorted(data_dirs)

    try:
        df_models = pd.read_excel(ft.model_list_excel)
    except Exception as e:
        logger.error(f"Failed to read model list Excel: {ft.model_list_excel} -> {e}")
        raise

    model_ids = df_models.get(ft.model_list_column, []).tolist()
    if not model_ids:
        logger.error(f"No models found in column '{ft.model_list_column}' of {ft.model_list_excel}")
        sys.exit(2)

    num_models = len(model_ids)
    total = len(data_dirs) * num_models
    logger.info(f"Found {len(data_dirs)} datasets x {num_models} models = {total} total jobs")

    assigned = []
    for di, data_dir in enumerate(data_dirs):
        for mi, model_id in enumerate(model_ids):
            linear_index = di * num_models + mi
            if linear_index % args.num_machines == args.machine_id:
                assigned.append((linear_index, data_dir, model_id))

    logger.info(f"Machine {args.machine_id} assigned {len(assigned)} jobs (of {total})")

    if args.dry_run:
        for idx, data_dir, model_id in assigned:
            print(f"[{idx}] {data_dir}  <--->  {model_id}")
        return

    # Run assigned jobs sequentially; users may wrap this script in a GPU-aware launcher
    for idx, data_dir, model_id in assigned:
        try:
            logger.info(f"RUN [{idx}]: {data_dir} / {model_id}")
            run_experiment(data_dir, model_id, ft.output_root, ft.seed, ft.batch_size)
        except Exception:
            logger.exception(f"FAILED [{idx}]: {data_dir} / {model_id}")


if __name__ == "__main__":
    main()
