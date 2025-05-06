from src.utils import load_data
import pickle
import os
from tqdm.auto import tqdm
from src.dataset import PersonaDataset,CollateFn,StateTrackingDistributedSampler
from torch.utils.data import DataLoader

file_path = os.path.join('data', 'train_data_name.pkl')
# data = load_data(file_path)


# # with open(file_path, 'rb') as f:
        
# #         data_set = pickle.load(f)

def main(rank):
    data_set = load_data(file_path)
    train_dataset = PersonaDataset(data_set)
    collate_fn = CollateFn(1)

    sampler = StateTrackingDistributedSampler(
        train_dataset, num_replicas=2, rank=rank,
        seed=42, checkpoint_file=f"sampler_rank{rank}.pkl"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn)

    # for batch in train_dataloader:
    #     print(len(train_dataloader))
    #     print(batch)
    #     sampler.save_checkpoint(0,sampler.batch_idx + 1)
    #     print(sampler.saved_indices[rank::2][:50])
    #     break
    for idx,batch in enumerate(tqdm(train_dataloader,desc="Loading"),start=sampler.batch_idx + 1):
        if idx % 1000 == 0:
            print(len(train_dataloader))
            # print(idx,batch)
            sampler.save_checkpoint(0,sampler.batch_idx + 1)
            print(sampler.saved_indices[rank::2][:50])
            break

main(0)