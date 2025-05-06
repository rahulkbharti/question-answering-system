import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# ==== 1. Load tokenizer with special tokens ====
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
tokenizer.add_special_tokens({
    'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']
})

# ==== 2. Load model ====
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
model.resize_token_embeddings(len(tokenizer))

# ==== 3. Load checkpoint from DDP-trained model ====
file_path = os.path.join('pretrained_models', 'model.pth')
checkpoint = torch.load(file_path, map_location="cpu")

# Remove 'module.' prefix if model was trained with DDP
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(new_state_dict)

# ==== 4. Set device and eval mode ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ==== 5. Chat loop with persona context ====
def chat_with_persona(model, tokenizer, device):
    print("Start chatting! Type 'exit' to stop.\n")

    persona_lines = [
        "I love playing video games. </s>",
        "Hey there, my name is Sidhartha and I am a veterinarian. </s>",
        "I am also a musician on the weekends. </s>",
        "Love to read drama books. </s>"
    ]
    persona = " ".join(persona_lines)
    context = f"<persona> {persona}"

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            break

        print("Thinking...")

        query = f"<query> {user_input}</s>"
        history.append(query)
        context += f" {query}"

        if len(history) > 5:
            context = context.replace(history[0], "", 1)
            history.pop(0)

        inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=1024).to(device)

        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print("Bot:", response)

        response_tagged = f"<response> {response}</s>"
        history.append(response_tagged)
        context += f" {response_tagged}"
        print("_" * 50)

# ==== 6. Start chat ====
chat_with_persona(model, tokenizer, device)
