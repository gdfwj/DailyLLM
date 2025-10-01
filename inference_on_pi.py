# -*- coding: utf-8 -*-
from pathlib import Path
import json, tqdm
from llama_cpp import Llama
from sklearn.metrics import classification_report, confusion_matrix

model_path = "/path/to/your/model/fuse_lora_ds-q6_k.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8
)

# use deepseek3 format
USER_TAG  = "<\uFF5CUser\uFF5C>"      
ASST_TAG  = "<\uFF5CAssistant\uFF5C>"
BOS       = "<s>"

def build_prompt(example):
    pieces = [BOS]
    for msg in example["messages"]:
        if msg["role"] in {"user", "system"}:
            pieces.append(f"{USER_TAG}{msg['content']}{ASST_TAG}")
    return "".join(pieces)
s
y_true, y_pred = [], []
data_path = Path("./test_data_status2016.jsonl")

for line in tqdm.tqdm(data_path.open()):
    sample = json.loads(line)
    prompt = build_prompt(sample)

    output = llm(
        prompt,
        max_tokens=8,            
        stop=[USER_TAG, ASST_TAG]  
    )["choices"][0]["text"].strip()

    y_true.append(sample["messages"][-1]["content"].strip())
    y_pred.append(output)

# for evaluation of classification tasks
print(confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))))
print(classification_report(y_true, y_pred, digits=4))
