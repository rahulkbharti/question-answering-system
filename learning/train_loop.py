import torch
from transformers import BartTokenizer

import torch.nn as nn
from . import BARTWithMemory  # Replace with the actual module where BARTWithMemory is defined

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Existing code
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BARTWithMemory().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

def train_step(batch):
    input_ids = batch["input_ids"].to(device)          # Dialogue history
    decoder_ids = batch["decoder_ids"].to(device)     # Response (shifted right)
    persona_ids = batch["persona_ids"].to(device)     # Persona sentences

    # Forward pass
    logits, ortho_loss = model(
        input_ids=input_ids,
        attention_mask=(input_ids != tokenizer.pad_token_id).float(),
        decoder_input_ids=decoder_ids[:, :-1],  # Exclude EOS
        persona_ids=persona_ids,
    )

    # Language modeling loss
    lm_loss = nn.CrossEntropyLoss()(
        logits.view(-1, model.config.vocab_size),
        decoder_ids[:, 1:].reshape(-1),  # Shifted left
    )

    # Total loss
    total_loss = lm_loss + model.ortho_loss_coeff * ortho_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()