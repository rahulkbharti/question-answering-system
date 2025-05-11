import os
import shutil
import torch
import math
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_polynomial_decay_schedule_with_warmup
import torch.distributed as dist

# From My Modules
from src.utils import load_data, load_model
from src.distributed_training import setup, cleanup
from src.dataset import PersonaDataset, CollateFn, StateTrackingDistributedSampler

# ==== Training Loop ====
def train_model(model, dataloader, optimizer, scheduler, rank, sampler, args=None, epoch=0, validation_dataloader=None):
    model.train()
    train_losses = []

    if rank == 0:
        tqdm_bar = tqdm(dataloader, desc="Training", initial=sampler.batch_idx+1, total=len(dataloader))
    else:
        tqdm_bar = dataloader

    for batch_idx, batch in enumerate(tqdm_bar, start=sampler.batch_idx):
        input_ids = batch['input_ids'].cuda(rank)
        decoder_input_ids = batch['decoder_input_ids'].cuda(rank)
        labels = batch['labels'].cuda(rank)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if rank == 0:
            ppx = math.exp(loss.item())
            tqdm_bar.set_postfix({'loss': f"{loss.item():.4f}", 'ppx': f"{ppx:.2f}" if not math.isinf(ppx) else "inf"})

        # Checkpoint saving logic
        if (batch_idx + 1) % args.checkpoint_interval == 0:
            sampler.save_checkpoint(epoch=epoch, batch_idx=batch_idx + 1)
            if rank == 0:
                latest_path = "./model_latest.pt"
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'start_epoch': epoch,
                }, latest_path)
                print(f"Checkpoint saved: {latest_path}")

    # Calculate global average loss and perplexity
    sum_loss = torch.tensor(sum(train_losses), dtype=torch.float32, device=rank)
    count = torch.tensor(len(train_losses), dtype=torch.int32, device=rank)
    dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_avg_loss = (sum_loss / count).item()
    global_ppx = math.exp(global_avg_loss)

    if rank == 0:
        print(f'Epoch {epoch} - Train loss: {global_avg_loss:.4f} | Train perplexity: {global_ppx:.2f}')

    # Validation
    if validation_dataloader is not None:
        val_loss, val_ppx = validate_model(model, validation_dataloader, rank)
        if rank == 0:
            print(f'Epoch {epoch} - Validation loss: {val_loss:.4f} | Validation perplexity: {val_ppx:.2f}')
            save_path = f"./model_epoch{epoch}.pt"
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, save_path)
            print(f"Model saved at {save_path}")


def validate_model(model, dataloader, rank):
    model.eval()
    validation_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=(rank != 0)):
            input_ids = batch['input_ids'].cuda(rank)
            labels = batch['labels'].cuda(rank)
            outputs = model(input_ids=input_ids, labels=labels)
            validation_losses.append(outputs.loss.item())

    sum_loss = torch.tensor(sum(validation_losses), dtype=torch.float32, device=rank)
    count = torch.tensor(len(validation_losses), dtype=torch.int32, device=rank)
    dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_avg_loss = (sum_loss / count).item()
    global_ppx = math.exp(global_avg_loss)

    return global_avg_loss, global_ppx

# ==== Main Training ====
def train(rank, args):
    setup(rank, args.world_size)
    
    # Load datasets
    train_data = load_data(args.data_path)
    valid_data = load_data(args.validation_path)
    train_dataset = PersonaDataset(train_data)
    valid_dataset = PersonaDataset(valid_data)

    collate_fn = CollateFn(1)  # Assuming pad_token_id=1

    # Samplers
    train_sampler = StateTrackingDistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank,
        seed=42, checkpoint_file=f"sampler_rank{rank}.pkl"
    )
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=collate_fn)

    # Model setup
    model = load_model(args.model_name).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_batches = len(train_loader)
    total_steps = args.num_epochs * num_batches
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, total_steps, power=2)

    # Load checkpoint
    start_epoch = 0
    checkpoint_path = "./model_latest.pt"
    if os.path.exists(checkpoint_path) and rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['start_epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        train_model(model, train_loader, optimizer, scheduler, rank, train_sampler, args, epoch, valid_loader)
    
    cleanup()