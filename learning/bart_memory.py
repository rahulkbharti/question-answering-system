import torch
from typing import Tuple, Optional
import torch.nn as nn
from transformers import BartForConditionalGeneration
from . import EntailmentMemory, DiscourseMemory

class BARTWithMemory(nn.Module):
    def __init__(self, bart_model_name: str = "facebook/bart-base", 
                 erm_slots: int = 10, ddm_slots: int = 5):
        super().__init__()
        # Load the full generation model to get lm_head
        bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart = bart.model  # The underlying BartModel
        self.lm_head = bart.lm_head  # Language modeling head
        self.final_logits_bias = bart.final_logits_bias  # Bias terms
        
        self.config = self.bart.config
        self.erm = EntailmentMemory(erm_slots, self.config.d_model)
        self.ddm = DiscourseMemory(ddm_slots, self.config.d_model)
        self.ortho_loss_coeff = 0.1

        # Special tokens
        self.sop_token_id = self.config.bos_token_id
        self.eop_token_id = self.config.eos_token_id
        self.soh_token_id = self.config.bos_token_id

    def orthogonal_loss(self) -> torch.Tensor:
        """Encourage ERM and DDM memories to be orthogonal."""
        return torch.sum(torch.matmul(self.erm.memory, self.ddm.memory.T) ** 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        persona_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode input
        encoder_outputs = self.bart.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        last_hidden = encoder_outputs.last_hidden_state

        # Get [z] token embedding (first token)
        z_token_embedding = last_hidden[:, 0, :]

        # Retrieve entailment (z) and discourse (z_d) representations
        z, _ = self.erm(z_token_embedding)
        z_d, _ = self.ddm(z_token_embedding)

        # Modify decoder's first token embedding
        decoder_inputs_embeds = self.bart.decoder.embed_tokens(decoder_input_ids)
        decoder_inputs_embeds[:, 0, :] += z + z_d

        # Decode
        decoder_outputs = self.bart.decoder(
            input_ids=None,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=last_hidden,
            attention_mask=attention_mask,
        )

        # Compute logits with lm_head and bias
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        ortho_loss = self.orthogonal_loss()

        return lm_logits, ortho_loss