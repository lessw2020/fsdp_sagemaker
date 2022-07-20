import os
import torch
import torch.distributed as dist
from typing import Dict, Union, Any, Tuple
from torch.utils.data import Dataset
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
import argparse
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import numpy as np
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)


from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


from vit_pytorch.deepvit import DeepViT, Residual
import gc
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
# from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import subprocess
# bfloat16 support verification imports (network and gpu native support)
import torch.cuda.nccl as nccl
from distutils.version import LooseVersion
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# for generation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)

from ChildTuningOptimizer import ChildTuningAdamW

# from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# local imports
import verify
import policies
from  grammar_dataset import get_dataset

# import datasets_grammar as dg
import tqdm

# config
import config
from utils.calculations_utils import calc_flop
import performance
os.environ["TRANSFORMERS_CACHE"] = '/tmp'

os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
# os.environ['NCCL_ALGO'] = 'RING'
# os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['RDMAV_FORK_SAFE'] = '1'
# os.environ['NCCL_MIN_NRINGS'] = '8'
# os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True

bf16_ready = (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
)

# save memory as gb
gb_unit_size = 1024 ** 3


def is_sm_run():
    return "TRAINING_JOB_NAME" in os.environ


def parse_args():

    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
    parser.add_argument("--save-dir", default="/model_chkpt", type=str)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 2022)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


def initialize_process_group(setup_args: Dict[str, Union[int, str]], backend: str = 'nccl') -> None:
    """
    Initialize process group.
    """
    master_addr, master_port = setup_args['master_addr'], setup_args['master_port']
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    args = {'backend': backend if torch.cuda.is_available() else 'gloo',
            'rank': setup_args['global_rank'],
            'world_size': setup_args['world_size']}
    dist.init_process_group(**args)


def get_setup_defaults(local_rank: int) -> Dict[str, Union[str, int]]:
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    world_size = get_num_nodes() * gpus_per_node
    node_rank = get_node_rank()
    global_rank = (node_rank * gpus_per_node) + local_rank
    print(f'local rank {local_rank} global rank {global_rank} world size {world_size}')
    ddp_setup_args = {'global_rank': global_rank,
                      'node_rank': node_rank,
                      'local_rank': local_rank,
                      'world_size': world_size,
                      'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
                      'master_port': '12355'}  # os.environ.get('MASTER_PORT', str(default_port))}
    return ddp_setup_args

def get_policies(cfg, local_rank):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.use_mixed_precision:
        bf16_ready = verify.bf16_ready

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if local_rank == 0:

                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if local_ == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    # wrapping policy -------
    # print(f"**overriding mp to fp16 - remove")
    # mixed_precision_policy = policies.fpSixteen

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy

def sync_all_device():
    # setup() has already configured CUDA_VISIBLE_DEVICES such that each
    # process exclusively works on its own set of devices. So it's safe to
    # do device sync here
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def train(
    args,
    model,
    local_rank,
    train_loader,
    optimizer,
    epoch,
    sampler=None,
    profiler=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        """print("************************")
        print(
            "train_loader",
            type(batch),
            batch["source_ids"].size(),
            batch["source_mask"].size(),
            batch["target_ids"].size(),
        )
        print("************************")
        """
        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )
        # print("##############################")
        # print(output.keys())
        # print("##############################")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
        if local_rank == 0:
            inner_pbar.update(1)
        if profiler:
            profiler.step()

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    train_accuracy = ddp_loss[0] / ddp_loss[1]
    if local_rank == 0:
        inner_pbar.close()

        print(
            f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
        )  # .format(epoch, train_accuracy))
    return train_accuracy


# ---- Validation ---------------


def validation(model, local_rank, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(local_rank)
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(test_loader)), colour="green", desc="r0 Validation Epoch"
        )
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            ddp_loss[0] += output["loss"].item()  # sum up batch loss
            ddp_loss[1] += len(batch)

            if local_rank == 0:
                inner_pbar.update(1)
            # pred = output.logits.argmax(
            #    dim=1, keepdim=True
            # )  # get the index of the max log-probability
            # ddp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
            # ddp_loss[2] += len(batch)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]

    if local_rank == 0:
        # test_loss = ddp_loss[0] / ddp_loss[1]
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def clear_gpu_cache(local_rank=None):
    print(f"clearing cache for rank {local_rank}")
    torch.cuda.empty_cache()

# _______________________________FSD_MAIN ______________________________________________
def run_fsdp(local_rank: int, *args: Any) -> None:
    cfg = config.benchmark_config()
    gpus_per_node = torch.cuda.device_count()
    
    world_size= get_num_nodes() * gpus_per_node

    print('/opt/amazon/efa/bin/fi_info')
    subprocess.run(["/opt/amazon/efa/bin/fi_info", "-p", "efa"])
    print('ls -l /dev/infiniband/uverbs0')
    subprocess.run(["ls", "-l", "/dev/infiniband/uverbs0"])
    
    fsdp_unit_params = cfg.fsdp_unit_size
    batch_size = cfg.batch_size
    if local_rank == 0:
        print(f"\n BatchSize = {batch_size}\n")

    val_batch_size = cfg.val_batch_size
    mp_policy, wrapping_policy = get_policies(cfg, fsdp_unit_params)

    if cfg.use_fp16:

        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler()



    # print_efa_info()
    # fsdp params count (min_num_params)
    fsdp_params_count_min = 20000

    # mixed precision policies

    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )


    args = args[0]
    setup_args = get_setup_defaults(local_rank=local_rank)
    initialize_process_group(setup_args)
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    print(f"finished init for local rank {local_rank}")
    model_name = cfg.model_name  # "google/t5-v1_1-small"  #   #
    save_name = model_name + "-"
    printable_model_name = str.replace(model_name, "/", "==")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    train_name = None
    if cfg.dataset_train:
        train_name = cfg.dataset_train

    train_dataset = get_dataset(tokenizer, train_name, 512, 512, True)
    if 0 == os.getenv("RANK"):
        print(f"--> Training Set Len = {len(train_dataset)}")
        print(f"using dataset {train_name}")
    # print("bailing")

    val_dataset = get_dataset(tokenizer, cfg.dataset_test, 512, 150, True)
    if 0 == os.getenv("RANK"):
        print(f"--> Validation set len = {len(val_dataset)}")
        print(f"using dataset {cfg.dataset_test}")

    sampler1 = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=local_rank, num_replicas=world_size)

    print(f"batch size = {batch_size}")
    # dataset = FakeDataset()
    # log_every = args.log_every
    # model = build_model(args.model_size)

    # model = model.to(torch.cuda.current_device())
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')

    print(f"batch size = {batch_size}")

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": val_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)


    if local_rank == 0:
        memmax = performance.Memory_Maximizer()
    if cfg.hf_activation_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"HF Activation checkpointing enabled\n")

    model_config = model.config
    embedding_size = (model.state_dict()["shared.weight"].shape)[1]
    FLOP = calc_flop(cfg, model_config, cfg.model_max_length, embedding_size)

    print(
        "embedding size **********",
        model.state_dict()["shared.weight"].shape,
        embedding_size,
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        # mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )

    # if torch.cuda.is_available():
    #     model.to(torch.cuda.current_device()) 

    # setting activation checkpointing 
    

    if cfg.fsdp_activation_checkpointing:
        policies.apply_fsdp_checkpointing(model)

    # setting optimizer 
    lr = 0.0008
    gamma = 0.85
    if cfg.use_task_free:
        optimizer = ChildTuningAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            reserve_p=cfg.percent_F,
            mode="taskfree",
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        if local_rank == 0:
            print(f"--> optimizer is AdamW")

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = cfg.num_epochs
    if local_rank == 0:
        print(f"Training for {epochs} epochs")
        
    best_train_accuracy = float("-inf")
    test_accuracy = float("-inf")
    curr_val_loss = float("inf")
    best_val_loss = float("inf")

    # --- main training loop - todo, this needs to be modularized
    if local_rank == 0:
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()
        memmax.start()

    torch_profiler = None

    if local_rank == 0 and cfg.track_memory:
        fn = cfg.model_name + "memory_tracking.txt"
        mem_alloc_tracker = []
        mem_reserved_tracker = []
    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        if local_rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        train_accuracy = train(
            args,
            model,
            local_rank,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
            profiler=torch_profiler,
        )
        if local_rank == 0:
            memmax.update()

        if cfg.run_validation:
            test_accuracy = validation(model, local_rank, test_loader)

        scheduler.step()

        if local_rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            total_epoch_time = time.time() - t0
            dur.append(total_epoch_time)

            train_acc_tracking.append(train_accuracy.item())
            print(f"TFLOP/s/GPU: {FLOP/10**12/total_epoch_time}")

            if cfg.run_validation:
                val_acc_tracking.append(test_accuracy.item())

            if cfg.track_memory:
                mem_alloc_tracker.append(torch.cuda.memory_allocated())
                mem_reserved_tracker.append(torch.cuda.memory_reserved())
                print(f"-->>>> reserved memroy in each epoch: {best_val_loss}")

        if local_rank == 0 and curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

    # sync_all_device()
    end_training_time = time.time()

    delays = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        delays, (end_training_time - start_training_time) / epochs
    )
    for i, item in enumerate(delays):
        delays[i] = round(item, 4)

    print("Flops cnt and delays", FLOP, delays)
    # tflops_gpu = FLOP / 10**12 * np.reciprocal(np.array(delays))
    if local_rank == 0:
        gflops_gpu = FLOP / 10**9 * np.reciprocal(np.array(delays))
        print(f"gflops per gpu={gflops_gpu}")
    # init_end_event.record()
    if local_rank == 0:
        # inner_pbar.close()
        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()

        # memory
        memmax.stop()
        if cfg.track_memory:
            print(f"total memory reserved: {mem_reserved_tracker}")
            print(f"total memory allocated: {mem_alloc_tracker}")

        print(f"Training accuracy: {train_acc_tracking}")
        if cfg.run_validation:
            print(f"Validation accuracy: {val_acc_tracking}")

        # memory summary
        if cfg.memory_report and local_rank == 0:
            print(
                f"CUDA Memory Summary After Last training:\n {torch.cuda.memory_summary()}"
            )

    dist.destroy_process_group()
    # dist.barrier()
    # cleanup()

def get_num_nodes() -> int:
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
        return len(cluster_inf['hosts'])
    return 1


def get_node_rank() -> int:
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
        return cluster_inf['hosts'].index(cluster_inf['current_host'])
    return 0




def print_efa_info():
    import subprocess
    print('/opt/amazon/efa/bin/fi_info')
    subprocess.run(["/opt/amazon/efa/bin/fi_info", "-p", "efa"])
    print('ls -l /dev/infiniband/uverbs0')
    subprocess.run(["ls", "-l", "/dev/infiniband/uverbs0"])

if __name__ == '__main__':
    if get_node_rank() == 0:
        import sys, shlex, subprocess

        cmd = shlex.split(
            f'{sys.executable} -m torch.utils.collect_env'
        )
        proc = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        print(proc.stdout)
    args = parse_args()
    gpus_per_machine = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(fn=run_fsdp,
             args=(args,),
             nprocs=gpus_per_machine,
             join=True)
