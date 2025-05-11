import os
import torch
import argparse
import torch.multiprocessing as mp

from src.utils import set_seed, open_config_file
from src.train import train

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Distributed Training Example')
    parser.add_argument('--config_path', type=str, default="config.yml", help='Path to the configuration file')

    args, _ = parser.parse_known_args()
    if os.path.exists(args.config_path):
        config = open_config_file(args.config_path)
    else:
        print(f"Warning: Configuration file {args.config_path} not found. Using default values.")
        config = {}

    parser.add_argument('--data_path', type=str, default=config.get('data_path', "/kaggle/input/train_data.pkl"), help='Path to the training data')
    parser.add_argument('--validation_path', type=str, default=config.get('validation_path', "/kaggle/input/train_data.pkl"), help='Path to the training data')
    parser.add_argument('--model_name', type=str, default=config.get('model_name', "facebook/bart-large"), help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 8), help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 1), help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate', 5e-5), help='Learning rate for the optimizer')
    parser.add_argument('--max_length', type=int, default=config.get('max_length', 512), help='Maximum sequence length for input data')
    parser.add_argument('--seed', type=int, default=config.get('seed', 42), help='Random seed for initialization')
    parser.add_argument('--output_dir', type=str, default=config.get('output_dir', "."), help='Directory to save the model and tokenizer')
    parser.add_argument('--log_dir', type=str, default=config.get('log_dir', "./logs"), help='Directory to save the logs')
    parser.add_argument('--checkpoint_dir', type=str, default=config.get('checkpoint_dir', "."), help='Directory to save the checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=config.get('checkpoint_interval', 1000), help='Interval for saving checkpoints')
    parser.add_argument('--warmup_ratio', type=float, default=config.get('warmup_ratio', 0.1), help='Warmup ratio for learning rate scheduler')

    args = parser.parse_args()


    '''
    # Actuall Logic Start From HereS
    
    '''

    set_seed(args.seed)  # Set seed for reproducibility
    
    world_size = torch.cuda.device_count()
    args.world_size = world_size
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    

    # print(args)
    # if world_size < 2:
    #     print("Need at least 2 GPUs for DDP.")
    #     return

    # Start the training process
    mp.spawn(train, args=(args,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()