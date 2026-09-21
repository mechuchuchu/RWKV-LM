# RWKV-7 Transformers trainer

`RWKV-v7/train_temp` is the original RWKV pretraining program. It expects a
native `rwkv-*.pth` checkpoint and a `binidx` token dataset. The model in
`RWKV7-G1j-1.5B-20260831` is a Transformers model split across safetensors, so
this directory provides a separate fine-tuning path for that format.

The trainer uses LoRA by default. This fits the supplied 1.5B model into a
12 GB GPU much more comfortably than full AdamW fine-tuning. Training records
can be either JSONL with a `text` field:

```json
{"text": "RWKV is a recurrent language model."}
```

or JSONL with a `messages` field:

```json
{"messages": [{"role": "user", "content": "What is RWKV?"}, {"role": "assistant", "content": "RWKV is a recurrent language model."}]}
```

Run it from the repository root:

```bash
source /venv/main/bin/activate
cd /workspace/rwkv-lm

python RWKV-v7/RWKV_trainer/train.py \
  --model_dir /workspace/.hf_home/hub/models--RWKV--RWKV7-G1j-1.5B-20260831/snapshots/2c18b29ab7fbece25ff6112281eea0fa41fcb30f \
  --train_file /workspace/data/train.jsonl \
  --output_dir /workspace/rwkv7-lora-out \
  --local_files_only \
  --max_length 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100
```

To use the adapted CUDA WKV kernel from the original RWKV training code, add
`--fused_wkv`. It requires BF16 CUDA training, pads sequence lengths internally
to a multiple of 16, and is intended for the trainer's zero-state training path:

```bash
python RWKV-v7/RWKV_trainer/train.py ... --fused_wkv
```

The output directory contains a PEFT adapter. Load it with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype="auto"
)
model = PeftModel.from_pretrained(base, "/workspace/rwkv7-lora-out")
tokenizer = AutoTokenizer.from_pretrained("/workspace/rwkv7-lora-out", trust_remote_code=True)
```

Use `--full_finetune` only when the GPU has enough memory. To train a different
set of linear layers, pass names such as `--lora_targets receptance key value
output`.
