import os
import sys
import subprocess
import argparse
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class CommandConfig:
    fdiri_cal: bool
    ref_genome: str

    outdir: str = None
    model_path: str = None
    model_config_path: str = None
    calibrator_path: str = None


def run_cmd(cmd):
    subprocess.call(cmd, shell=True)


def get_experiment_model_name(path: str, identify: str = 'Train') -> str:
    normalized_path = os.path.normpath(path)
    path_parts = normalized_path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part.startswith(identify):
            return path_parts[i - 1]
    raise ValueError(f"The provided path does not contain a '{identify}' directory.")


def check_file_exists(outdir, test_data_path):
    opt_test_name = os.path.splitext(os.path.basename(test_data_path))[0]
    opt_pred_file = os.path.join(outdir, f'{opt_test_name}.tsv.gz')
    return os.path.exists(opt_pred_file)

def create_command(config: CommandConfig, test_data_path: str,
                   cuda_id: Optional[int] = None, pred_batch_size: int = 128):
    outdir = config.outdir
    opt_test_name = os.path.splitext(os.path.basename(test_data_path))[0]
    opt_pred_file = os.path.join(outdir, f'{opt_test_name}.tsv.gz')

    # if opt_pred_file , return None
    if os.path.exists(opt_pred_file):
        return None

    cmd = (
        f'mural_snv predict --ref_genome {config.ref_genome} '
        f'--test_data {test_data_path} --model_path {config.model_path} '
        f'--model_config_path {config.model_config_path} '
        f'--kmer_corr 3 5 7 --region_corr 10000 50000 --segment_center 10000 '
        f'--pred_file {opt_pred_file} '
    )

    if config.fdiri_cal:
        cmd += f'--calibrator_path {config.calibrator_path} '
    else:
        cmd += '--save_each_model_preds '

    if cuda_id is not None:
        cmd += f'--cuda_id {cuda_id} --pred_batch_size {pred_batch_size} '
    else:
        cmd += '--cpu_only '

    cmd += f'> {outdir}/{opt_test_name}.out 2> {outdir}/{opt_test_name}.err'
    return cmd


def dispatch_tasks(command_config: CommandConfig, test_data_list: List[str],
                   gpu_ids: Optional[List[int]] = None, tasks_per_gpu: int = 20,
                   cpu_workers: int = 0, pred_batch_size: int = 128):
    if not test_data_list:
        print("No tasks to run.")
        return

    if not gpu_ids:
        gpu_ids = []

    total_gpu_capacity = len(gpu_ids) * tasks_per_gpu

    # 根据是否有 CPU workers 决定是否允许溢出
    if cpu_workers > 0:
        gpu_data = test_data_list[:total_gpu_capacity]
        cpu_data = test_data_list[total_gpu_capacity:]
    elif gpu_ids:
        # 无 CPU workers，全部任务走 GPU（忽略 tasks_per_gpu 上限）
        gpu_data = test_data_list
        cpu_data = []
    else:
        # 无 GPU，全部走 CPU（此情况已被 parse_args 拦截，防御性处理）
        gpu_data = []
        cpu_data = test_data_list

    # 轮询分配到各 GPU
    gpu_cmds_by_device = {gid: [] for gid in gpu_ids}
    for i, data_path in enumerate(gpu_data):
        gid = gpu_ids[i % len(gpu_ids)]
        cmd = create_command(command_config, data_path, cuda_id=gid,
                             pred_batch_size=pred_batch_size)
        if cmd is None:
            continue
        gpu_cmds_by_device[gid].append(cmd)

    cpu_cmds = [
        create_command(command_config, data_path, cuda_id=None)
        for data_path in cpu_data
    ]
    cpu_cmds = [cmd for cmd in cpu_cmds if cmd is not None]

    total_gpu_tasks = sum(len(v) for v in gpu_cmds_by_device.values())
    total_cpu_tasks = len(cpu_cmds)
    print(f"Task dispatch: {total_gpu_tasks} GPU tasks "
          f"({', '.join(f'GPU{g}:{len(v)}' for g, v in gpu_cmds_by_device.items())}), "
          f"{total_cpu_tasks} CPU tasks ({cpu_workers} workers)")

    if total_gpu_tasks == 0 and total_cpu_tasks == 0:
        print("Warning: no tasks dispatched.")
        return

    futures = []
    gpu_executors = []

    for gid in gpu_ids:
        cmds = gpu_cmds_by_device[gid]
        if not cmds:
            continue
        executor = ProcessPoolExecutor(max_workers=tasks_per_gpu)
        gpu_executors.append(executor)
        for cmd in cmds:
            futures.append(executor.submit(run_cmd, cmd))

    cpu_executor = None
    if cpu_cmds and cpu_workers > 0:
        cpu_executor = ProcessPoolExecutor(max_workers=cpu_workers)
        for cmd in cpu_cmds:
            futures.append(cpu_executor.submit(run_cmd, cmd))

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Task failed: {e}")

    for executor in gpu_executors:
        executor.shutdown(wait=False)
    if cpu_executor:
        cpu_executor.shutdown(wait=False)

    print("All tasks completed.")

def collect_test_data(input_dir: str, outdir: str) -> List[str]:
    """Collect .gz test data files, skipping already-predicted ones."""
    all_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir) if f.endswith('.gz')
    ])
    pending = [f for f in all_files if not check_file_exists(outdir, f)]
    return pending


def process_test_data(command_config: CommandConfig, check_path: str,
                      input_dir: str, output_dir: str,
                      gpu_ids: Optional[List[int]] = None, tasks_per_gpu: int = 20,
                      cpu_workers: int = 0, pred_batch_size: int = 128):
    """
    Run parallel predictions on all .gz files in input_dir.

    Args:
        command_config: Model and calibration settings.
        check_path: Checkpoint directory containing model, model.config.pkl,
                     and model.fdiri_cal.pkl.
        input_dir: Directory containing .gz test data files.
        output_dir: Directory for prediction outputs.
        gpu_ids: List of GPU device IDs. None for CPU-only.
        tasks_per_gpu: Max concurrent tasks per GPU.
        cpu_workers: Number of CPU workers for overflow tasks.
        pred_batch_size: Batch size for GPU inference.
    """
    command_config.model_path = os.path.join(check_path, 'model')
    command_config.model_config_path = os.path.join(check_path, 'model.config.pkl')
    command_config.calibrator_path = os.path.join(check_path, 'model.fdiri_cal.pkl')

    if not command_config.fdiri_cal:
        output_dir = f'{output_dir}_no_fdiri'
    command_config.outdir = output_dir

    os.makedirs(output_dir, exist_ok=True)

    test_data_list = collect_test_data(input_dir, output_dir)
    print(f"{len(test_data_list)} tasks pending in {input_dir}")

    dispatch_tasks(
        command_config, test_data_list,
        gpu_ids=gpu_ids, tasks_per_gpu=tasks_per_gpu,
        cpu_workers=cpu_workers, pred_batch_size=pred_batch_size
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parallel prediction dispatcher for mural_snv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dual GPU, 20 concurrent tasks per GPU
  %(prog)s --ref_genome hg38.fa --check_path /path/to/ckpt \\
      --input_dir /data/chr5/split --output_dir results/chr5 --gpu_ids 0 1

  # Single GPU + 8 CPU overflow workers
  %(prog)s --ref_genome hg38.fa --check_path /path/to/ckpt \\
      --input_dir /data/chr5/split --output_dir results/chr5 \\
      --gpu_ids 0 --cpu_workers 8

  # CPU only, 16 workers
  %(prog)s --ref_genome hg38.fa --check_path /path/to/ckpt \\
      --input_dir /data/chr5/split --output_dir results/chr5 --cpu_workers 16

  # Custom batch size, disable fdiri calibration
  %(prog)s --ref_genome hg38.fa --check_path /path/to/ckpt \\
      --input_dir /data/chr5/split --output_dir results/chr5 \\
      --gpu_ids 0 --no_fdiri --pred_batch_size 256
        """
    )

    # Required
    parser.add_argument('--ref_genome', required=True,
                        help='Reference genome file path')
    parser.add_argument('--check_path', required=True,
                        help='Checkpoint directory (must contain model, '
                             'model.config.pkl, model.fdiri_cal.pkl)')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing .gz test data files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for prediction output files')

    # GPU/CPU scheduling
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                        help='GPU device IDs (e.g. --gpu_ids 0 1). Omit for CPU-only.')
    parser.add_argument('--tasks_per_gpu', type=int, default=20,
                        help='Max concurrent tasks per GPU (default: 20)')
    parser.add_argument('--cpu_workers', type=int, default=0,
                        help='CPU worker count for overflow tasks (default: 0)')
    parser.add_argument('--pred_batch_size', type=int, default=128,
                        help='Batch size for GPU inference (default: 128)')

    # Model options
    parser.add_argument('--no_fdiri', action='store_true',
                        help='Disable fdiri calibration')

    args = parser.parse_args()

    if not args.gpu_ids and args.cpu_workers <= 0:
        parser.error('Must specify at least --gpu_ids or --cpu_workers > 0')

    return args


def main():
    args = parse_args()

    command_config = CommandConfig(
        fdiri_cal=not args.no_fdiri,
        ref_genome=args.ref_genome,
    )

    print(f"Config: {command_config}")
    print(f"GPUs: {args.gpu_ids}, tasks_per_gpu: {args.tasks_per_gpu}, "
          f"cpu_workers: {args.cpu_workers}, batch_size: {args.pred_batch_size}")

    process_test_data(
        command_config, args.check_path,
        args.input_dir, args.output_dir,
        gpu_ids=args.gpu_ids, tasks_per_gpu=args.tasks_per_gpu,
        cpu_workers=args.cpu_workers, pred_batch_size=args.pred_batch_size
    )


if __name__ == "__main__":
    main()