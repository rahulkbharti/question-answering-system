import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random  # <-- Add this import


# ==== Custom MNIST Model ====
def build_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )


# ==== Build Train & Val Dataloaders ====
def build_dataloaders(rank, world_size):
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler)

    return train_loader, val_loader, train_sampler


# ==== Training Loop ====
def train_loop(rank, model, train_loader, train_sampler, optimizer, criterion, val_loader, epochs):
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(rank), labels.cuda(rank)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        validate(rank, model, val_loader)


# ==== Validation Function ====
def validate(rank, model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(rank), labels.cuda(rank)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if rank == 0:
        accuracy = 100.0 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%\n")


# ==== DDP Setup ====
def setup_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# ==== Training Entry ====
def train(rank, world_size):
    if rank >= torch.cuda.device_count():
        print(f"Invalid rank {rank} for available GPUs.")
        return

    setup_process_group(rank, world_size)

    model = build_model().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_loader, val_loader, train_sampler = build_dataloaders(rank, world_size)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loop(rank, ddp_model, train_loader, train_sampler, optimizer, criterion, val_loader, epochs=10)

    # === Save model only from rank 0 ===
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/mnist_ddp_model.pth")
        print("Model saved to checkpoints/mnist_ddp_model.pth")

    cleanup()


# ==== Main Spawn ====
def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP needs at least 2 GPUs.")
        return

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()