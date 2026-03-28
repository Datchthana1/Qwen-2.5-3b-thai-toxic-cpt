import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-3B"
ADAPTER_PATH = "./qwen2.5-cpt-thai-toxic/checkpoint-52"

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
model.eval()

print("Model ready!\n")


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.1, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    print("=== Thai Toxic Tweet CPT Model ===")
    print("พิมพ์ข้อความเพื่อให้ model ต่อ | พิมพ์ 'quit' เพื่อออก\n")

    while True:
        prompt = input("Input: ").strip()

        if prompt.lower() == "quit":
            break
        if not prompt:
            continue

        output = generate(prompt)
        print(f"Output: {output}\n")
