
import os
import torch
import pickle

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data.distributed import DistributedSampler

# ==== Custom Collate Function ====
class CollateFn:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        decoder_input_ids = [item['decoder_input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # ignore index

        return {
            'input_ids': input_ids,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }

# ==== Custom Dataset for CNNDailyMail ====
class PersonaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'])
        # attention_mask = torch.tensor(item['attention_mask'])
        labels = torch.tensor(item['labels'])
         

        decoder_input_ids = labels
        # labels = labels[1:]

        # decoder_input_ids = labels[:-1]
        # labels = labels[1:]

        return {
            'input_ids': input_ids,
            # 'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }
   
# Custom DistributedSampler with checkpointing
class StateTrackingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=42, checkpoint_file=None):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        self.checkpoint_file = checkpoint_file or f"sampler_rank{rank}.pkl"
        self.epoch = 0
        self.batch_idx = 0
        self.saved_indices = None
        self._load_checkpoint()

    def __iter__(self):
        if self.saved_indices is None:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            self.saved_indices = indices
        else:
            indices = self.saved_indices

        indices = indices[self.rank::self.num_replicas]
        return iter(indices[self.batch_idx:])

    def __len__(self):
        return len(self.dataset) // self.num_replicas

    def save_checkpoint(self, epoch, batch_idx):
        self.epoch = epoch
        self.batch_idx = batch_idx
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({
                'epoch': self.epoch,
                'batch_idx': self.batch_idx,
                'indices': self.saved_indices
            }, f)
        print(f"✅ Rank {self.rank}: Saved checkpoint at epoch {epoch}, batch {batch_idx}")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
                self.epoch = state['epoch']
                self.batch_idx = state['batch_idx']
                self.saved_indices = state['indices']
                print(f"🔁 Rank {self.rank}: Restored epoch {self.epoch}, batch {self.batch_idx}")
        else:
            self.epoch = 0
            self.batch_idx = 0
            self.saved_indices = None
