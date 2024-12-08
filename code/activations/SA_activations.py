import json
from huggingface_hub import login
import os
import transformers
import torch
import numpy as np
from tqdm import tqdm


with open('../../data/test_oversight.jsonl', 'r') as file:
    test_data = [json.loads(line) for line in file]
with open('../../data/deploy_oversight.jsonl', 'r') as file:
    deploy_data = [json.loads(line) for line in file]


login(os.getenv('HF_TOKEN'))

quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct" # MODEL
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


batch_size = 8
layer = model.model.layers[-1].mlp.act_fn

def create_hook_fn(activation_list):
    def hook_fn(module, input, output):
        final_tokens = output[:, -1, :].detach()
        for token in final_tokens:
            activation_list.append(token.cpu().to(torch.float32).numpy())
        return output
    return hook_fn


test_data_activations = []
hook = layer.register_forward_hook(create_hook_fn(test_data_activations))

data_batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]

for batch in tqdm(data_batches, desc="Processing test data batches"):
    batch = [
        "<|start_header_id|>system<|end_header_id|>\r\n"
        "\r\n"
        "Cutting Knowledge Date: December 2023\r\n"
        "Today Date: 1 January 2024\r\n"
        "\r\n"
        "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\r\n"
        "\r\n"
        f"{item['body']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        for item in batch
    ]         
    batch_inputs = tokenizer(batch,
                             padding=True,
                             truncation=True,
                             return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**batch_inputs)

test_data_activations_np = np.array(test_data_activations)
np.save('../../results/test_data_activations_70B_quant.npy', test_data_activations_np) # SAVE

hook.remove()


deploy_data_activations = []
hook = layer.register_forward_hook(create_hook_fn(deploy_data_activations))

data_batches = [deploy_data[i:i + batch_size] for i in range(0, len(deploy_data), batch_size)]

for batch in tqdm(data_batches, desc="Processing deploy data batches"):
    batch = [f"USER: {item['body']} ASSISTANT:" for item in batch]
    batch_inputs = tokenizer(batch,
                             padding=True,
                             truncation=True,
                             return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**batch_inputs)

deploy_data_activations_np = np.array(deploy_data_activations)
np.save('../../results/deploy_data_activations_70B_quant.npy', deploy_data_activations_np) # SAVE

hook.remove()