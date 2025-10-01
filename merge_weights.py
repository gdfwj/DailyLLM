import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = "/path/to/deepseek"
lora = "/path/to/lora"
out  = "/path/to/merged"

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, lora, device_map="auto", torch_dtype=torch.float16)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
