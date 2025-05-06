import os
import shutil
import torch
import math
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_polynomial_decay_schedule_with_warmup


# From My Modules
from src.utils import load_data, load_model
from src.distributed_training import setup, cleanup
from src.dataset import PersonaDataset, CollateFn, StateTrackingDistributedSampler




# ==== Training Loop ====
def train_model(model, dataloader, optimizer,scheduler, rank,sampler,args=None,epoch=0):
    # Set the model to training mode
    model.train()
    train_losses = []
    train_perplexity = []

    tqdm_bar = tqdm(dataloader, desc=f"Rank {rank} Training", initial=sampler.batch_idx+1, total=len(dataloader))
    for batch_idx, batch in enumerate(tqdm_bar, start=sampler.batch_idx):
        # Move the batch to the appropriate device
        input_ids = batch['input_ids'].cuda(rank)
        decoder_input_ids = batch['decoder_input_ids'].cuda(rank)
        labels = batch['labels'].cuda(rank)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.detach())
        ppx = torch.exp(loss.detach())
        train_perplexity.append(ppx)
        
        tqdm_bar.set_postfix({
           'loss': f"{loss.item():.4f}",
           'ppx': f"{ppx.item():.2f}" if not math.isinf(ppx.item()) else "inf"
        }) 
        
        # if rank == 0:  # Log only from the main process to avoid redundant logging
        #     print(f"GPU {rank} - Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item()}")
           

        # ==== Checkpoint saving logic ====
        if (batch_idx + 1) % args.checkpoint_interval == 0:
            latest_path = f"./model_latest_gpu{rank}.pt"
            # previous_path = f"{args.checkpoint_dir}/model_previous_gpu{rank}.pt"

            # # If a latest checkpoint exists, move it to previous
            # if os.path.exists(latest_path):
            #     shutil.move(latest_path, previous_path)
            
            # Save the new model as latest
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'start_epoch': epoch,
            }, latest_path)
            # print(f"Model checkpoint updated — latest: {latest_path}, previous: {previous_path}")
            print(f"Model checkpoint updated — latest: {latest_path}")
            sampler.save_checkpoint(epoch=epoch, batch_idx=batch_idx + 1)


    train_losses = [loss.item() for loss in train_losses]
    train_perplexity = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in train_perplexity]
    train_loss = np.mean(train_losses)
    train_ppx = np.mean(train_perplexity)
    print(f'Train loss: {train_loss}  | Train perplexity: {train_ppx}')


# ==== Main Training ====
def train(rank, args):
    # Setup the process group for distributed training
    setup(rank, args.world_size)
    
    # Load the dataset
    data_set = load_data(args.data_path)
    train_dataset = PersonaDataset(data_set)
    collate_fn = CollateFn(1)

    sampler = StateTrackingDistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank,
        seed=42, checkpoint_file=f"sampler_rank{rank}.pkl"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    
    # # Load the model
    model = load_model(args.model_name)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    # # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # # Calculate total training steps
    num_batches = len(train_dataloader)
    total_train_steps = args.num_epochs* num_batches
    warmup_steps = int(args.warmup_ratio * total_train_steps)

    scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            power=2
    )

    # # Load checkpoint if available
    start_epoch = 0
    checkpoint_path = f"./model_latest_gpu{rank}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['start_epoch']
        print(f"Resumed from checkpoint at epoch {start_epoch}")
        # logger.info(f"Resumed from checkpoint at batch {start_batch}")
    

    # Start the training process
    for epoch in range(start_epoch,args.num_epochs):
        sampler.set_epoch(epoch)
        train_model(model, train_dataloader, optimizer,scheduler,rank,sampler,args)
    
    # Cleanup the model and optimizer

    cleanup()
