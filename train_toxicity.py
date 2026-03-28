import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig , 
    get_peft_model , 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("tmu-nlp/thai_toxicity_tweet", trust_remote_code=True)

def format_raw_text(dataset):
    return {
        "text": dataset['tweet_text']
        }

dataset = dataset.map(format_raw_text)
dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)

model_id = "Qwen/Qwen2.5-3B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True ,
    bnb_4bit_quant_type="nf4" ,
    bnb_4bit_compute_dtype=torch.bfloat16 ,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16 ,
    lora_alpha=32 ,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ] ,
    lora_dropout=0.05 ,
    bias = "none" ,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir="./qwen2.5-cpt-thai-toxic" ,
    num_train_epochs=3 ,
    per_device_train_batch_size=1 ,
    gradient_accumulation_steps=16 ,
    gradient_checkpointing=True ,

    optim="paged_adamw_8bit" ,
    learning_rate=5e-5 ,
    lr_scheduler_type="cosine" ,
    warmup_ratio=0.1 ,

    bf16=True ,
    max_length=256 ,

    logging_steps=10 ,
    save_strategy="epoch" ,
    eval_strategy="epoch" ,

    dataset_text_field="text" ,
    packing=True
)

trainer = SFTTrainer(
    model = model ,
    args=training_args ,
    train_dataset=dataset['train'] ,
    eval_dataset=dataset['test']
)

trainer.train()
model.save_pretrained("./cpt-lora-adapter")

# --------------------------------------------------------------------> BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,           # โหลด model ด้วย 4-bit quantization (ลด VRAM ~4x)
#                                  # ถ้า False = ไม่ quantize (ต้องการ VRAM เต็ม)
#     bnb_4bit_quant_type="nf4",   # วิธี quantize
#                                  # "nf4" = NormalFloat4 (แนะนำ, เหมาะกับ weight ที่กระจายแบบ normal)
#                                  # "fp4" = Float4 (ทั่วไปกว่า แต่ accuracy ต่ำกว่า nf4 เล็กน้อย)
#     bnb_4bit_compute_dtype=torch.bfloat16,  # dtype ที่ใช้ตอน compute (ไม่ใช่ตอน store)
#                                              # torch.bfloat16 = แนะนำสำหรับ GPU รุ่นใหม่ (Ampere+)
#                                              # torch.float16  = รองรับ GPU เก่ากว่า แต่เสี่ยง overflow
#                                              # torch.float32  = ช้าและกิน VRAM มากที่สุด
#     bnb_4bit_use_double_quant=True  # Quantize ตัว quantization constant อีกรอบ (ประหยัด VRAM ~0.4 bit/param)
#                                     # True = ประหยัด VRAM มากขึ้น, False = ปิดฟีเจอร์นี้
# )

# --------------------------------------------------------------------> LoraConfig
# lora_config = LoraConfig(
#     r=32,            # Rank ของ matrix ใน LoRA (ขนาดของ adapter)
#                      # ยิ่งสูง = จำนวน trainable param มากขึ้น, เรียนรู้ได้ซับซ้อนขึ้น แต่กิน VRAM มากขึ้น
#                      # ค่าทั่วไป: 8, 16, 32, 64
#                      # งานง่าย → r=8~16, งานซับซ้อน → r=32~64
#     lora_alpha=64,   # Scaling factor สำหรับ LoRA weight
#                      # ผล = lora_alpha / r (เรียกว่า effective scale)
#                      # นิยมตั้ง = r*2 หรือ r เช่น r=32 → alpha=64 หรือ 32
#                      # ยิ่งสูง = LoRA weight มีอิทธิพลต่อ model มากขึ้น
#     target_modules=[ # Layer ไหนบ้างที่ใส่ LoRA adapter
#         "q_proj", "k_proj", "v_proj", "o_proj",      # Attention layers
#         "gate_proj", "up_proj", "down_proj"           # FFN layers
#     ],               # ใส่ครบทุก layer = เรียนรู้ได้มากขึ้น แต่ช้าและกิน VRAM มากขึ้น
#                      # ใส่แค่ q_proj, v_proj = เร็วกว่า ประหยัดกว่า (แบบดั้งเดิม)
#     lora_dropout=0.05,  # Dropout rate ใน LoRA layer (ป้องกัน overfitting)
#                         # 0.0 = ปิด dropout (ถ้า dataset ใหญ่)
#                         # 0.05~0.1 = ค่าทั่วไป
#                         # ยิ่งสูง = regularize มากขึ้น แต่ train ช้าขึ้น
#     bias="none",     # Train bias ไหนบ้าง (อธิบายไปแล้วในข้อก่อน)
#                      # "none" / "lora_only" / "all"
#                       "none" — ใช้ในกรณีส่วนใหญ่
#                           Parameter น้อยที่สุด, เร็วที่สุด
#                           ผลลัพธ์มักไม่ต่างจาก lora_only มากนัก
#                       "lora_only" — ถ้าอยากให้ model ปรับตัวได้มากขึ้นนิดหน่อย
#                           เพิ่ม parameter เล็กน้อย (~ไม่กี่ KB)
#                           บางงานได้ผลดีกว่า none เล็กน้อย
#                       "all" — แทบไม่ค่อยใช้
#                           Parameter เพิ่มขึ้นมาก
#                           เสี่ยง overfit และ checkpoint ใหญ่ขึ้น
#     task_type="CAUSAL_LM"  # ประเภทงาน (โค้ดมี typo: task_task → task_type)
#                             # "CAUSAL_LM"       = Text generation (GPT-style)
#                             # "SEQ_2_SEQ_LM"    = Encoder-Decoder (T5-style)
#                             # "SEQ_CLS"         = Text classification
#                             # "TOKEN_CLS"       = Token classification (NER)
#                             # "QUESTION_ANS"    = QA
# )

# --------------------------------------------------------------------> SFTConfig 
# training_args = SFTConfig(
#     output_dir="./qwen2.5-cpt-thai-toxic",  # โฟลเดอร์บันทึก checkpoint และ model
#     num_train_epochs=2,             # จำนวนรอบที่วน train ผ่านข้อมูลทั้งหมด
#                                     # น้อยไป = underfit, มากไป = overfit
#                                     # ทั่วไป: 1~5 epochs
#     per_device_train_batch_size=1,  # จำนวน sample ต่อ batch ต่อ GPU
#                                     # ยิ่งมาก = เร็วขึ้น แต่กิน VRAM มากขึ้น
#                                     # GPU VRAM น้อย → ใช้ 1~2
#     gradient_accumulation_steps=16, # สะสม gradient กี่ step ก่อน update weight ครั้งนึง
#                                      # effective batch size = batch_size × accum_steps = 1×16 = 16
#                                      # ใช้แทนการเพิ่ม batch_size เมื่อ VRAM ไม่พอ
#     gradient_checkpointing=True,    # คำนวณ gradient ใหม่แทนการเก็บไว้ใน memory
#                                     # True = ประหยัด VRAM ~30~40% แต่ช้าลง ~20%
#                                     # False = เร็วกว่า แต่กิน VRAM มากกว่า
#     optim="paged_adamw_8bit",       # Optimizer ที่ใช้
#                                     # "paged_adamw_8bit"  = AdamW แบบ 8-bit + page memory (แนะนำสำหรับ QLoRA)
#                                     # "adamw_8bit"        = AdamW 8-bit (ไม่มี paging)
#                                     # "adamw_torch"       = AdamW ปกติ (กิน VRAM มากสุด)
#                                     # "sgd"               = SGD (เร็วแต่ผลมักแย่กว่า Adam)
#     learning_rate=1e-4,             # ความเร็วในการเรียนรู้
#                                     # สูงไป = diverge, ต่ำไป = train นาน/ไม่ converge
#                                     # LoRA ทั่วไป: 1e-4 ~ 3e-
#     lr_scheduler_type="cosine",     # วิธีปรับ learning rate ระหว่าง train
#                                     # "cosine"  = ค่อยๆ ลดแบบ cosine curve (แนะนำ)
#                                     # "linear"  = ลดตรงๆ
#                                     # "constant"= คงที่ตลอด
#                                     # "cosine_with_restarts" = cosine แล้ว reset หลายรอบ
#     warmup_ratio=0.1,               # สัดส่วนของ total steps ที่ค่อยๆ เพิ่ม lr จาก 0 ขึ้นมา
#                                     # 0.1 = 10% แรกของ training เป็น warmup
#                                     # ช่วยให้ model ไม่ explode ตอนเริ่มต้น
#     bf16=True,                      # ใช้ bfloat16 precision ตอน train
#                                     # True  = เร็วขึ้น ประหยัด VRAM (ต้องการ Ampere GPU+)
#                                     # False = ใช้ float32 (ช้ากว่า กิน VRAM มากกว่า)
#                                     # fp16=True = อีกทางเลือก แต่เสี่ยง overflow มากกว่า bf16
#     max_length=256,                 # ความยาว token สูงสุดต่อ sample
#                                     # ยิ่งมาก = รองรับข้อความยาวขึ้น แต่กิน VRAM มากขึ้น (quadratic ใน attention)
#                                     # ควรดูจาก dataset ว่าข้อความยาวแค่ไหน
#     logging_steps=10,               # log metric (loss ฯลฯ) ทุกกี่ step
#                                     # น้อย = เห็น progress บ่อย แต่ overhead มากขึ้น
#     save_strategy="epoch",          # บันทึก checkpoint เมื่อไหร่
#                                     # "epoch"  = บันทึกทุก epoch
#                                     # "steps"  = บันทึกทุก N steps (ต้องตั้ง save_steps ด้วย)
#                                     # "no"     = ไม่บันทึกระหว่าง train
#     eval_strategy="epoch",          # ประเมินผลเมื่อไหร่ (ตัวเลือกเหมือน save_strategy)
#     dataset_text_field="text",      # ชื่อ column ใน dataset ที่เป็น input text
#                                     # ต้องตรงกับที่ map ไว้ใน format_raw_text()
#     packing=True                    # รวม sample สั้นๆ หลายชิ้นให้เต็ม max_length ใน 1 batch
#                                     # True  = train เร็วขึ้นมากสำหรับ dataset ที่มี sample สั้น
#                                     # False = แต่ละ sample อยู่ใน batch ของตัวเอง
# )
