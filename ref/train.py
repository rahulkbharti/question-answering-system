import os
import torch
import pickle
import yaml
import shutil
import argparse
import math
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
from datetime import datetime
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data.distributed import DistributedSampler


# ==== Set Seed for Reproducibility ====
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==== Setup Logger ====
log_path = os.path.join("./output", "training.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==== Oprn Config File ====
def open_config_file(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        logger.error(f"Configuration file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        return {}

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
class CNNDailyMailDataset(Dataset):
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

# ==== Setup Process Group ====
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# ==== Cleanup Process Group ====
def cleanup():
    dist.destroy_process_group()

# ==== Load Data ====
def load_data(data_path):
    print(f"Loading data from {data_path}")
    logger.info(f"Loading data from {data_path}")

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        logger.error(f"Data file {data_path} not found.")
        return None
    
    with open(data_path, 'rb') as f:
        data_set = pickle.load(f)
    return data_set

# ==== Load Tokenizer and Model ====
def load_model(model_name):
    print(f"Loading model {model_name}")
    logger.info(f"Loading model {model_name}")
    # Load the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}
    tokenizer.add_special_tokens(add_special_tokens)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model

# ==== Training Loop ====
def train_model(model, dataloader, optimizer,scheduler, rank, start_batch=0, args=None):
    # Set the model to training mode
    model.train()
    train_losses = []
    train_perplexity = []

    for batch_idx, batch in enumerate(dataloader, start=start_batch):
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
        
        if rank == 0:  # Log only from the main process to avoid redundant logging
            print(f"GPU {rank} - Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item()}")
            logger.info(f"GPU {rank} - Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item()}")
    
        # ==== Checkpoint saving logic ====
        if (batch_idx + 1) % args.checkpoint_interval == 0:
            latest_path = f"{args.checkpoint_dir}/model_latest_gpu{rank}.pt"
            previous_path = f"{args.checkpoint_dir}/model_previous_gpu{rank}.pt"

            # If a latest checkpoint exists, move it to previous
            if os.path.exists(latest_path):
                shutil.move(latest_path, previous_path)

            # Save the new model as latest
            torch.save({
                'epoch': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss.item()
            }, latest_path)
            print(f"Model checkpoint updated — latest: {latest_path}, previous: {previous_path}")
            logger.info(f"Model checkpoint updated — latest: {latest_path}, previous: {previous_path}")

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
    train_dataset = CNNDailyMailDataset(data_set)
    collate_fn = CollateFn(1)

    sampler = StateTrackingDistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank,
        seed=42, checkpoint_file=f"sampler_rank{rank}.pkl"
    )
    # sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    
    # Load the model
    model = load_model(args.model_name)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Calculate total training steps
    num_batches = len(train_dataloader)
    total_train_steps = args.num_epochs* num_batches
    warmup_steps = int(args.warmup_ratio * total_train_steps)

    scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            power=2
    )

    # Load checkpoint if available
    checkpoint_path = f"{args.checkpoint_dir}/model_latest_gpu{rank}.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_batch = checkpoint['epoch']  # Use 'epoch' to get the last processed batch
        logger.info(f"Resumed from checkpoint at batch {start_batch}")
    else:
        start_batch = 0
        logger.info(f"No checkpoint found at {checkpoint_path}, starting fresh training.")
    
    # Start the training process
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        train_model(model, train_dataloader, optimizer,scheduler,rank,start_batch,args)
    
    # Cleanup the model and optimizer
    cleanup()

# ==== Multiprocessing Spawn ====
def main():

    parser = argparse.ArgumentParser(description='Distributed Training Example')

    parser.add_argument('--config_path', type=str, default="config.yml", help='Path to the configuration file')

    args, _ = parser.parse_known_args()
    if os.path.exists(args.config_path):
        config = open_config_file(args.config_path)
    else:
        print(f"Warning: Configuration file {args.config_path} not found. Using default values.")
        logger.warning(f"Configuration file {args.config_path} not found. Using default values.")
        config = {}

    parser.add_argument('--data_path', type=str, default=config.get('data_path', "/kaggle/input/train_data.pkl"), help='Path to the training data')
    parser.add_argument('--model_name', type=str, default=config.get('model_name', "facebook/bart-large"), help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 4), help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 1), help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate', 5e-5), help='Learning rate for the optimizer')
    parser.add_argument('--max_length', type=int, default=config.get('max_length', 512), help='Maximum sequence length for input data')
    parser.add_argument('--seed', type=int, default=config.get('seed', 783435), help='Random seed for initialization')
    parser.add_argument('--output_dir', type=str, default=config.get('output_dir', "./output"), help='Directory to save the model and tokenizer')
    parser.add_argument('--log_dir', type=str, default=config.get('log_dir', "./logs"), help='Directory to save the logs')
    parser.add_argument('--checkpoint_dir', type=str, default=config.get('checkpoint_dir', "./checkpoints"), help='Directory to save the checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=config.get('checkpoint_interval', 1000), help='Interval for saving checkpoints')
    parser.add_argument('--warmup_ratio', type=float, default=config.get('warmup_ratio', 0.1), help='Warmup ratio for learning rate scheduler')
    args = parser.parse_args()
    set_seed(args.seed)  # Set seed for reproducibility
    
    
#     # Setup logging
#     logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s | %(levelname)s | %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     handlers=[
#         logging.FileHandler(args.log_dir + f"/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#         logging.StreamHandler()
#     ]
#    )
#     global logger
#     logger = logging.getLogger(__name__)


    world_size = torch.cuda.device_count()
    args.world_size = world_size
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    
    if world_size < 2:
        print("Need at least 2 GPUs for DDP.")
        return

    # Start the training process
    mp.spawn(train, args=(args,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()