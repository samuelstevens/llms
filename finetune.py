"""
We want to use decoder-only language models to do seq2seq problems rather than just language modeling. To tune an LM on a seq2seq problem, we don't care about its loss over the input, only on the output.

To tune these large models, we use FullyShardedDataParallel (FSDP) from PyTorch to split the model over multiple GPUs efficiently.

For simplicity, we use sequences padded at the end. In the future, we could use NestedTensors which apparently work, but since I am just starting with FullyShardedDataParallel, I want to leave that for future work.
"""

import heapq
import json
import logging
import os

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import modeling
import wandb

model_name = "llama-7b"
tokenizer_ckpt = "/research/nfs_su_809/workspace/shared/llama/raw/tokenizer.model"
model_ckpt = "/research/nfs_su_809/workspace/shared/llama/raw/7B/consolidated.00.pth"
data_path = "data/databricks-dolly-15k.jsonl"  # about 3.7M tokens, 7.5 MB

# Training
global_batch_size = 4
gradient_accumulation_steps = 2
learning_rate = 3e-4
weight_decay = 0.01
grad_clip = 2.0
epochs = 300
warmup_steps = 1000

dtype = "float16"
# PyTorch dtype
ptdtype = getattr(torch, dtype)
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
# Don't need scaler for bfloat16 because it's higher range, lower precision.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

save_root = "/local/scratch/stevens.994/llm/checkpoints"
log_every = 10
n_latest_checkpoints = 3
n_best_checkpoints = 2
best_metric = "train/loss"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------
# Distributed setup
# -----------------

is_ddp = int(os.environ.get("LOCAL_RANK", -1)) != -1
if is_ddp:
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_master = rank == 0
    seed_offset = rank
    torch.distributed.barrier()
else:
    rank = 0
    world_size = 1
    is_master = True
    seed_offset = 0

device = f"cuda:{rank}"
torch.cuda.set_device(device)

local_batch_size = global_batch_size // world_size // gradient_accumulation_steps
step = 0

# -------------------
# Logging and metrics
# -------------------

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(f"Rank {rank}")


def make_pair(*, instruction, context, response, category):
    if context:
        return (
            f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n# Instruction:\n{instruction}\n\n# Input:\n{context}\n\n# Response:\n",
            response,
        )
    else:
        return (
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n# Instruction:\n{instruction}\n\n# Response:\n",
            response,
        )


def build_dataset():
    """
    Prepare the data. It's only 30 MB as int64 tokens, so we could just leave
    the entire dataset on the GPU, but we have to pad each batch, which requires
    making a new tensor on the GPU. Because of this, we leave the tensors on CPU and
    use multiprocessing with torch dataloaders to efficiently load data on GPUs.
    """
    dataset = []
    with open(data_path) as fd:
        for line in fd:
            prompt, answer = make_pair(**json.loads(line))

            ctx = tokenizer.encode(prompt, bos=True, eos=False, out="pt")
            out = tokenizer.encode(answer, bos=False, eos=True, out="pt")

            example = torch.cat((ctx, out))
            x, y = example[:-1], example[1:]
            # mask needs to be same shape as y, and ignores the bos token
            mask = torch.cat((torch.zeros_like(ctx), torch.ones_like(out)))[1:]

            dataset.append((x, y, mask))

    return dataset


def build_dataloader():
    def padded_collate_fn(batch) -> dict[str, object]:
        B = len(batch)
        L = max(len(x) for (x, y, m) in batch)
        xs = torch.full((B, L), model_cfg.pad_id, dtype=torch.int64)
        ys = torch.full((B, L), model_cfg.pad_id, dtype=torch.int64)
        mask = torch.zeros_like(ys)
        for i, (x, y, m) in enumerate(batch):
            xs[i][: len(x)] = x
            ys[i][: len(y)] = y
            mask[i][: len(m)] = m

        return dict(toks=xs, targets=ys, loss_mask=mask)

    dataset = build_dataset()

    if is_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            drop_last=False,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=8,
        # TODO: Evaluate whether this helps throughput on A100 and on A6000.
        pin_memory=True,
        persistent_workers=True,
        collate_fn=padded_collate_fn,
    )


# --------
# Training
# --------


def get_lr():
    # Linear warmup followed by constant LR.
    if step < warmup_steps:
        return learning_rate * step / warmup_steps

    return learning_rate


def move(obj: object, device) -> object:
    if hasattr(obj, "to"):
        return obj.to(device)

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[move(key, device)] = move(value, device)
        return obj
    elif isinstance(obj, list):
        return [move(elem, device) for elem in obj]
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(type(obj))


def train():
    logger.info(f"Starting epoch {epoch} training.")
    global step
    model.train()

    for i, batch in enumerate(tqdm(dataloader)):
        # Whether we will actually do an optimization step
        will_step = i % gradient_accumulation_steps == 0
        # Only need to sync if we're doing an optimization step
        model.require_backward_grad_sync = will_step

        batch = move(batch, device)

        with ctx:
            logits, loss = model(**batch)

        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        # Update the learning rate while we wait for the forward pass
        lr = get_lr()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        grad = None
        if will_step:
            # Clip gradients (TODO: update for FSDP: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_)
            # if grad_clip > 0:
                # scaler.unscale_(optimizer)
                # grad = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            # This has to be after optimizer.step() or scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            step += 1

        if i % (log_every * gradient_accumulation_steps) == 0 and is_master:
            # Need to use **{} because train/lr isn't a valid variable name
            # We can use loss.item() because we don't need to sync it across
            # all processes since it's going to be noisy anyways.
            metrics = {
                "train/lr": lr,
                "train/loss": loss.item(),
                "train/step": step,
                "perf/total-examples": step * global_batch_size,
                "perf/batches": i,
                "train/grad": grad,
            }
            wandb.log(metrics)

    logger.info(f"Finished epoch {epoch} training.")


def save():
    if not is_master:
        return

    logger.info("Saving checkpoint.")
    ckpt = {
        # TODO: include training loss in here
        "model": model_without_ddp.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    directory = os.path.join(save_root, run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"ep{epoch}.pt")
    torch.save(ckpt, path)
    logger.info("Saved to %s.", path)

    # Prune checkpoints: keep n_best_checkpoints and n_latest_checkpoints
    # ------------------

    # Load all checkpoints and select which ones to keep.
    all_ckpts = {path: ckpt}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        all_ckpts[path] = torch.load(path, map_location="cpu")

    # Get the best and latest checkpoints
    best_ckpts = heapq.nlargest(
        n_best_checkpoints, all_ckpts, key=lambda c: all_ckpts[c][best_metric]
    )
    best_ckpts = set(best_ckpts)
    latest_ckpts = heapq.nlargest(
        n_latest_checkpoints, all_ckpts, key=lambda c: all_ckpts[c]["epoch"]
    )
    latest_ckpts = set(latest_ckpts)

    # If not one of the best or latest, remove it
    for path in all_ckpts:
        if path in best_ckpts:
            logger.info(
                "Keep %s. It's in the top %d for %s.",
                path,
                n_best_checkpoints,
                best_metric,
            )
            continue

        if path in latest_ckpts:
            logger.info("Keep %s. It's in the %d latest.", path, n_latest_checkpoints)
            continue

        os.remove(path)
        logger.info("Removed %s.", path)


def restore():
    # Restores the latest checkpoint.
    latest_ckpt = None
    directory = os.path.join(save_root, run_id)
    if not os.path.isdir(directory):
        raise RuntimeError(f"No checkpoint directory at {directory}.")

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        ckpt = torch.load(path, map_location=device)
        if not latest_ckpt or ckpt["epoch"] > latest_ckpt["epoch"]:
            latest_ckpt = ckpt

    if not latest_ckpt:
        raise RuntimeError(f"No saved checkpoints in {directory}.")

    global model, optimizer, start, step
    model_without_ddp.load_state_dict(latest_ckpt["model"])
    optimizer.load_state_dict(latest_ckpt["optim"])
    # Add one because we save after we finish an epoch
    start = latest_ckpt["epoch"] + 1
    step = latest_ckpt["step"]


if __name__ == "__main__":
    # First epoch
    start = 0
    
    model_cfg = modeling.Config(n_layer=4, n_head=8, n_embd=1024)
    # model_cfg = modeling.Config.from_name(model_name)
    # model = modeling.load_pretrained_llama(model_cfg, model_ckpt)
    model = modeling.Llama(model_cfg)
    tokenizer = modeling.load_pretrained_tokenizer(model_cfg, tokenizer_ckpt)
    model_without_ddp = model
    model.to(device)
    # TODO: try compiling

    if is_ddp:
        # TODO: needs to be FSDP
        # model = torch.distributed.fsdp.FullyShardedDataParallel(
            # model
        # )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    dataloader = build_dataloader()
    # The optimizer must be initialized after the module has been wrapped, since FSDP will shard parameters in-place and this will break any previously initialized optimizers.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
    )

    resumed_and_id = [False, None]
    if is_master:
        run = wandb.init(
            project="llm-instruct",
            entity="samuelstevens",
            config=dict(
                model=model_name,
                # data options
                data_path=data_path,
                gradient_accumulation_steps=gradient_accumulation_steps,
                # training options
                epochs=epochs,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                amp=dtype,
                grad_clip=grad_clip,
                debug=False,
            ),
            job_type="finetune",
            resume=False,
        )
        resumed_and_id = run.resumed, run.id

    # Now non-master processes have resumed and run_id
    # Refer to https://github.com/pytorch/pytorch/issues/56142
    # for why we need a variable instead of an anonymous list
    if is_ddp:
        torch.distributed.broadcast_object_list(resumed_and_id)
    resumed, run_id = resumed_and_id

    if resumed:
        assert run_id is not None, "If we resume, we need a run.id"
        restore()

    logger.info("Starting training from epoch %s.", start)
    for epoch in range(start, epochs):
        train()
        save()

    logger.info("Done.")
