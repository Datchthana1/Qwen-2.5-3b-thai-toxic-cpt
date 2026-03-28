---
language:
  - th
license: apache-2.0
base_model: Qwen/Qwen2.5-3B
tags:
  - lora
  - qlora
  - peft
  - thai
  - causal-lm
  - continued-pretraining
datasets:
  - tmu-nlp/thai_toxicity_tweet
---

# Qwen2.5-3B — Thai Toxic Tweet CPT (LoRA)

Continued Pre-Training (CPT) ของ [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) ด้วย LoRA บน dataset ภาษาไทย [`tmu-nlp/thai_toxicity_tweet`](https://huggingface.co/datasets/tmu-nlp/thai_toxicity_tweet)

> **หมายเหตุ:** โมเดลนี้ไม่ใช่ toxicity classifier — เป็นการ adapt model ให้เรียนรู้สไตล์และรูปแบบภาษาของ Thai tweet

---

## การใช้งาน

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-3B"
ADAPTER_REPO  = "Datchthana/qwen2.5-thai-toxic-cpt"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()

def generate(prompt, max_new_tokens=256, temperature=0.8, top_p=0.9):
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
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

print(generate("รัฐบาลชุดนี้มัน..."))
```

> **Tips:** prompt ที่ค้างกลางประโยคหรือมี `...` ต่อท้ายจะได้ผลดีที่สุด

---

## Training Details

| Parameter | ค่า |
|-----------|-----|
| Base Model | Qwen/Qwen2.5-3B |
| Method | QLoRA (4-bit NF4) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o/gate/up/down proj |
| Epochs | 3 |
| Learning rate | 5e-5 |
| LR Scheduler | cosine |
| Effective batch size | 16 |
| Max length | 256 tokens |
| Optimizer | paged_adamw_8bit |
| Framework | transformers + peft + trl |

---

## Training Results

| Epoch | Train Loss | Eval Loss | Token Accuracy |
|-------|-----------|-----------|----------------|
| 1 | 3.059 | 2.864 | 41.9% |
| 2 | 2.718 | 2.737 | 44.4% |
| 3 | 2.721 | — | 44.6% |

---

## ข้อจำกัด

- เป็น **next token prediction** เท่านั้น ไม่ใช่ chatbot และไม่ตอบคำถาม
- ประโยคสมบูรณ์จะได้ output ว่าง เพราะ model generate EOS ทันที
- dataset มีภาษาหยาบและเนื้อหา sensitive — ใช้ด้วยความระมัดระวัง
