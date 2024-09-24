from datasets import load_dataset
import numpy as np
from huggingface_hub import login
import os
import transformers
import torch
from tqdm import tqdm


harmful_data = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
CAV = torch.from_numpy(np.load('results/model_parameters_70B_quant.npz')['coefficients']) # CAV


login(os.getenv('HF_TOKEN'))

quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct" # MODEL
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config # QUANTIZATION
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


layer = model.model.layers[-1].mlp.act_fn

def create_hook_fn(dir_derivs_example):
    def hook_fn(module, grad_input, grad_output):
        gradient = grad_output[0].squeeze(0)[-1].detach().to(torch.float32)
        dir_deriv = torch.dot(gradient, CAV.to(gradient.device))
        dir_derivs_example.append(dir_deriv.cpu())
    return hook_fn


dir_derivs_examples = {}
generated_tokens_examples = {}
for i, example in tqdm(enumerate(harmful_data['harmful']['Goal']), desc="Processing harmful data"):
    example = f"USER: {example} ASSISTANT:"
    dir_derivs_example = []
    generated_tokens_example = []
    hook = layer.register_full_backward_hook(create_hook_fn(dir_derivs_example))

    input = tokenizer(example, return_tensors="pt").to(model.device)

    for _ in range(25): # PREDICTED TOKENS
        model.zero_grad()
        logits = model(**input).logits.squeeze(0)[-1]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(probabilities)
        log_prob = torch.log(probabilities[next_token])        
        input['input_ids'] = torch.cat([input['input_ids'], next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        log_prob.backward()
        generated_tokens_example.append(tokenizer.decode(next_token.item()))

    dir_derivs_examples[f'example_{i}'] = dir_derivs_example
    generated_tokens_examples[f'example_{i}'] = generated_tokens_example

    hook.remove()

np.savez('results/gradients_70B_quant.npz', **dir_derivs_examples) # SAVE
np.savez('results/generated_tokens_70B_quant.npz', **generated_tokens_examples) # SAVE

