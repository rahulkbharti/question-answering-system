import torch
from torch.utils.data import DataLoader
from modules.dataset import PersonaDataset, CollateFn, StateTrackingDistributedSampler
from modules.utils import load_data
import os
from tqdm import tqdm

def main(rank=0, world_size=2):
    # 1. Load your data
    file_path = os.path.join('data', 'train_data_name.pkl')
    dataset = PersonaDataset(load_data(file_path))

    # 2. Initialize your sampler (will load checkpoint automatically)
    sampler = StateTrackingDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=42,
        checkpoint_file=f"sampler_rank{rank}.pkl"
    )

    # 3. Use DataLoader with your custom sampler
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=CollateFn(pad_token_id=1),  # replace 1 with your actual PAD token ID
        drop_last=False
    )

    # 4. Resume training loop from saved batch index
    tqdm_bar = tqdm(dataloader, desc=f"Rank {rank} Training", initial=sampler.batch_idx+1, total=len(dataloader))
    for idx, batch in enumerate(tqdm_bar, start=sampler.batch_idx):
        # === Training Step Here ===
        print(f"Rank {rank} | Batch {idx} | Input IDs shape: {batch['input_ids'].shape}")

        # 5. Save checkpoint every N steps
        if (idx + 1) % 50 == 0:
            sampler.save_checkpoint(epoch=0, batch_idx=idx + 1)
            break  # For demo purposes, stop after one checkpoint

if __name__ == "__main__":
    main(rank=1, world_size=2)  # or use torch.multiprocessing for multi-rank
