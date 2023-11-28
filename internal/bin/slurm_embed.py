from typing import Optional
import subprocess
import time
import shutil
from pathlib import Path
import submitit
import typer

# Requing requires time to be specified as number
DAY = 60 * 24


def embed(
    *,
    passages: str,
    output_dir: str,
    model_name_or_path: str,
    shard_id: int,
    num_shards: int,
):
    args = [
        "python",
        "generate_passage_embeddings.py",
        "--model_name_or_path",
        model_name_or_path,
        "--output_dir",
        output_dir,
        "--passages",
        passages,
        "--shard_id",
        f"{shard_id}",
        "--num_shards",
        f"{num_shards}",
    ]
    subprocess.run(args, check=True)


def main(
    passages: str = "/checkpoint/par/replug/psgs_w100.tsv",
    output_dir: str = "/checkpoint/par/replug/output/wiki_embed",
    model_name_or_path: str = "facebook/contriever",
    log_dir: str = "/checkpoint/par/slurm/replug",
    days: int = 3,
    num_shards: int = 4 * 8,
    partition="devlab",
    cluster: Optional[str] = None,
    mem="64G",
    gres="gpu:1",
    cpu_per_task: int = 8,
):
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    executor = submitit.AutoExecutor(folder=log_dir, cluster=cluster)
    executor.update_parameters(
        cpus_per_task=cpu_per_task,
        nodes=1,
        slurm_ntasks_per_node=1,
        slurm_gres=gres,
        slurm_time=days * DAY,
        slurm_job_name="replug-embed",
        slurm_partition=partition,
        slurm_mem=mem,
        slurm_constraint="volta32gb,ib4",
    )
    print("Submitting job")
    jobs = []
    with executor.batch():
        for shard_id in range(num_shards):
            job = executor.submit(
                embed,
                passages=passages,
                output_dir=output_dir,
                model_name_or_path=model_name_or_path,
                shard_id=shard_id,
                num_shards=num_shards,
            )
            jobs.append(job)

    if shutil.which("squeue") is not None:
        print("Waiting 5 seconds to print: squeue --me")
        time.sleep(5)
        subprocess.run(
            "squeue --me --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R'",
            shell=True,
        )
    print("Waiting for completion")
    for j in jobs:
        print(j.result())


if __name__ == "__main__":
    typer.run(main)
