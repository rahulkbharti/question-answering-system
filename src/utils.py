import os
import pickle
import yaml
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# ==== Set Seed for Reproducibility ====
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==== Oprn Config File ====
def open_config_file(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
       
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
       
        return {}

# ==== Load Data ====
def load_data(data_path):
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return None
    
    with open(data_path, 'rb') as f:
        data_set = pickle.load(f)
    return data_set

# ==== Load Tokenizer and Model ====
def load_model(model_name):
    print(f"Loading model {model_name}")

    # Load the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}
    tokenizer.add_special_tokens(add_special_tokens)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model

