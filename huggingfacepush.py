import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login, HfApi

BASE_MODEL_ID = "Qwen/Qwen2.5-3B"
ADAPTER_PATH = "./qwen2.5-cpt-thai-toxic/checkpoint-52"
HF_REPO = "Datchthana/qwen2.5-thai-toxic-cpt"

login()  # จะ prompt ให้ใส่ HF token

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Pushing to Hugging Face Hub...")
model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)

api = HfApi()
api.upload_file(
    path_or_fileobj="./README.md",
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="model"
)

print(f"Done! https://huggingface.co/{HF_REPO}")
