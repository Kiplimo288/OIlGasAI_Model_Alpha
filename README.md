# OilGasAI Model Alpha

[![License](https://img.shields.io/badge/license-Llama%203.1%20Community-blue)](LICENSE)
[![Model](https://img.shields.io/badge/🤗%20Model-OilGasAI--Model--Alpha-yellow)](https://huggingface.co/OilgasAI/OilGasAI-Model-Alpha)
[![Platform](https://img.shields.io/badge/Platform-OilGasAI-green)](https://innovations1-spec.github.io/oilgasai/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)

**OilGasAI Model Alpha** is a domain-expert large language model purpose-built for the oil and gas industry. It is a QLoRA fine-tune of Meta's Llama 3.1 70B Instruct, trained on 21,429 domain-specific instruction-response pairs using compute resources provided by UNDP and CINECA's Leonardo HPC supercomputer in Bologna, Italy.

This repository contains the full inference stack, RAG pipeline, FastAPI backend, and sensor integration layer that powers the [OilGasAI platform](https://innovations1-spec.github.io/oilgasai/) — an AI-driven system for emissions monitoring, regulatory compliance, and predictive maintenance in oil and gas operations.

---

## The Story Behind the Model

OilGasAI Model Alpha grew directly out of field research. The foundation was laid at Colorado State University's METEC facility as part of the SABER project, where the work centered on calibrating low-cost methane sensors — Raspberry Pi-based measurement units running Gaussian plume dispersion models and random forest sensor correction algorithms, deployed in real controlled-release emission environments.

That hands-on sensor work exposed a deeper problem: even when field operators had good measurements, translating raw sensor data into EPA-compliant reports was slow, error-prone, and required expensive regulatory expertise that most small and mid-size operators simply don't have access to. The gap wasn't in the sensing — it was in turning data into compliance.

OilGasAI Model Alpha is built to close that gap. The model brings together domain knowledge from EPA regulations, peer-reviewed methane research, SCADA engineering documentation, and real sensor calibration data — making it accessible through a conversational AI interface integrated directly into the OilGasAI emissions compliance platform.

The model was trained with computing support from the **UNDP AI Hub for Sustainable Development Compute Accelerator Programme** (10,000 GPU hours) and **CINECA's ISCRA programme** (80,000 GPU hours on the Leonardo supercomputer) — making it one of the most compute-resourced domain LLMs built specifically for energy emissions compliance.

The low-cost sensors from the original METEC/SABER field deployments — MIRA methane sensors, Figaro TGS2611, and Alphasense OB3 units — are now directly integrated into the platform's sensor calibration layer, closing the loop from field measurement to AI-assisted regulatory reporting.

---

## What It Does

| Capability | Description |
|---|---|
| **EPA GHGRP Compliance** | Subpart W reporting, CDX submission guidance, emissions factor calculations |
| **Methane Monitoring** | Low-cost sensor calibration, drift correction, LDAR |
| **Predictive Maintenance** | SCADA-based compressor monitoring, failure prediction |
| **Regulatory Guidance** | NSPS OOOOa/b/c, OGMP 2.0 Level 1–5, state-level regulations |
| **Report Auto-Generation** | Automated compliance report drafting from sensor data |
| **RAG Pipeline** | 37,009 vector FAISS index for document-grounded responses |

---

## Model Architecture

| Attribute | Detail |
|---|---|
| Base Model | meta-llama/Llama-3.1-70B-Instruct |
| Architecture | 80-layer decoder-only transformer, GQA (8 KV / 64 Q heads), RoPE, SwiGLU |
| Hidden Dimension | 8192 |
| Vocabulary | 128K tokens |
| Fine-Tuning Method | QLoRA (4-bit NF4 quantization) |
| Trainable Parameters | 828,375,040 (1.1% of full 70B) |
| LoRA Adapter Size | 1.6 GB |
| License | Llama 3.1 Community License |

---

## How the Model Was Built — Step by Step

### Step 1: Data Collection & Curation

The training corpus consists of **21,429 instruction-response pairs** drawn from seven source categories, each targeting a critical competency area in upstream and midstream oil and gas:

| Source | Count | Content Focus |
|---|---|---|
| ArXiv Papers | ~5,000 | Methane detection, sensor calibration, emissions modeling |
| PubMed Abstracts | ~4,000 | Environmental health, gas exposure, air quality |
| EPA Regulations | ~3,000 | GHGRP Subpart W, NSPS OOOOa/b/c, compliance procedures |
| O&G Engineering Docs | ~4,000 | Compressor maintenance, SCADA analytics, process optimization |
| Sensor Calibration Data | ~2,000 | Low-cost sensor drift correction, reference method comparison |
| OGMP 2.0 Framework | ~1,500 | Level 1–5 reporting, measurement-based quantification |
| Synthetic Q&A | ~1,929 | Identity training, edge cases |

An 80/20 train/validation split produced **17,143 training** / **4,286 validation** examples.

---

### Step 2: Data Preprocessing & Formatting

All pairs were converted into Llama 3.1 chat template format with system, user, and assistant roles. Max sequence length was enforced at 2,048 tokens; samples exceeding the context window were discarded. The final dataset was saved in JSONL format for streaming into the trainer.

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are OilGasAI Model Alpha, a domain expert for oil and gas...
<|start_header_id|>user<|end_header_id|>
{user_query}
<|start_header_id|>assistant<|end_header_id|>
{assistant_response}<|eot_id|>
```

---

### Step 3: Infrastructure — Leonardo HPC at CINECA

Training ran on CINECA's Leonardo supercomputer in Bologna, Italy — one of Europe's top HPC systems — under an ISCRA allocation of 80,000 GPU-hours, supported by the UNDP AI Hub Compute Accelerator Programme.

Each compute node: **4× NVIDIA A100 64GB SXM4**, interconnected with HDR200 InfiniBand.

| Component | Version |
|---|---|
| Python | 3.11.7 |
| PyTorch | 2.x |
| Transformers | 5.1.0 |
| PEFT | 0.12.0+ |
| TRL | 0.28.0 |
| BitsAndBytes | 0.49.1 |
| Accelerate | 1.12.0 |
| CUDA | 12.x |

---

### Step 4: Stage 1 — Domain Fine-Tuning (QLoRA, ~24 hours)

The base Llama 3.1 70B Instruct model was loaded in 4-bit NF4 quantization. Low-rank adapters were inserted into all seven attention and feedforward projection layers and trained with TRL's SFTTrainer across 4 GPUs via Accelerate.

| Parameter | Value |
|---|---|
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Batch size | 2 per device × 4 GPUs |
| Gradient accumulation | 4 steps |
| Effective batch size | 32 |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup steps | 30 |
| Max sequence length | 2,048 |
| Optimizer | AdamW (paged, 8-bit) |
| Training steps | ~800 |
| **Training loss** | **1.90 → 1.08** |

---

### Step 5: Stage 2 — Identity Fine-Tuning (~1 hour)

A second, shorter stage reinforced the OilGasAI Model Alpha identity so the model consistently identifies itself across all interactions. This used a curated mix of 750 identity examples ("Who are you?", "What can you do?") and 500 domain examples to avoid catastrophic forgetting, trained for 3 epochs at learning rate 1e-4.

---

### Step 6: Evaluation & Validation

Evaluated on 4,286 held-out validation examples plus manual expert assessment across six domain categories.

#### Domain Evaluation Results

| Category | Rating | Notes |
|---|---|---|
| EPA Subpart W Reporting | ⭐⭐⭐⭐⭐ | Accurate emissions factors, calculation methods, reporting deadlines |
| Sensor Calibration | ⭐⭐⭐⭐⭐ | Correct drift correction, reference gas procedures |
| Compressor Maintenance | ⭐⭐⭐⭐⭐ | Proper SCADA parameter identification, failure mode analysis |
| OGMP 2.0 Compliance | ⭐⭐⭐⭐ | Good framework knowledge; RAG helps for Level 5 specifics |
| Identity Consistency | ⭐⭐⭐⭐⭐ | Consistent OilGasAI identity across all prompts |
| Domain Boundaries | ⭐⭐⭐⭐ | Appropriately redirects off-topic queries |

#### RAG Pipeline Results (37,009 vectors, FAISS + BGE-large-en-v1.5)

| Query Type | RAG Score | Response Quality |
|---|---|---|
| EPA GHGRP Subpart W | 0.847 | Excellent |
| Low-cost sensor calibration | 0.823 | Excellent |
| Compressor predictive maintenance | 0.812 | Excellent |
| OGMP Level 5 reporting | 0.688 | Good (needs more source docs) |

---

### Step 7: Packaging & Publication

The LoRA adapter weights (1.6 GB) were published to Hugging Face as `OilgasAI/OilGasAI-Model-Alpha`. The adapter is distributed separately from the base model — users load Llama 3.1 70B Instruct and apply the OilGasAI adapter on top.

---

### Step 8: Deployment & Integration

OilGasAI Model Alpha is deployed as part of a **multimodal AI platform** with three integrated components:

- **OilGasAI Model Alpha** — Text-based domain expert, loaded in 4-bit quantization
- **Vision Module (Qwen2-VL-7B)** — SCADA screenshot analysis, PFD interpretation, equipment photo inspection
- **RAG Pipeline (37,009 vectors)** — FAISS index with BGE-large-en-v1.5 for document-grounded responses

The sensor calibration layer directly integrates field-deployed low-cost methane sensors — turning raw IoT readings into EPA-compliant LDAR reports automatically, completing the loop from sensor to compliance.

---

## End-to-End Pipeline Summary

| Step | Phase | Duration | Output |
|---|---|---|---|
| 1 | Data Collection & Curation | Weeks | 21,429 instruction-response pairs |
| 2 | Data Preprocessing & Formatting | Days | JSONL dataset in Llama 3.1 chat format |
| 3 | Infrastructure Setup | Hours | Leonardo HPC environment ready |
| 4 | Stage 1: Domain Fine-Tuning | ~24 hours | Domain-adapted LoRA checkpoint |
| 5 | Stage 2: Identity Fine-Tuning | ~1 hour | Identity-reinforced LoRA checkpoint |
| 6 | Evaluation & Validation | Days | Eval report, RAG benchmark scores |
| 7 | Packaging & Publication | Hours | HuggingFace: OilgasAI/OilGasAI-Model-Alpha |
| 8 | Deployment & Integration | Days | FastAPI multimodal server live |

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OilGasAI Platform                         │
│          https://innovations1-spec.github.io/oilgasai/      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend (api/)                       │
│  /chat  |  /compliance/report  |  /sensors/calibrate        │
└──────┬──────────────┬───────────────────┬────────────────────┘
       │              │                   │
┌──────▼──────┐  ┌────▼──────┐   ┌───────▼────────────────┐
│  inference/ │  │   rag/    │   │   sensors/              │
│  Model Alpha│  │  FAISS +  │   │  MIRA, Figaro TGS2611,  │
│  LoRA 1.6GB │  │  BGE-large│   │  Alphasense OB3 + LDAR  │
└─────────────┘  └───────────┘   └────────────────────────┘
```

---

## Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/oilgasai.git
cd oilgasai
```

### 2. Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Add your HF_TOKEN — request model access at https://huggingface.co/OilgasAI/OilGasAI-Model-Alpha
```

### 4. Run
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are EPA Subpart W reporting requirements for compressors?"}'
```

---

## No GPU? Use the HF Inference API

```bash
export USE_HF_API=true
export HF_TOKEN=your_token_here
uvicorn api.main:app --reload
```

---

## Loading the Model Directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "OilgasAI/OilGasAI-Model-Alpha")
tokenizer = AutoTokenizer.from_pretrained("OilgasAI/OilGasAI-Model-Alpha")

messages = [{"role": "user", "content": "What are EPA Subpart W reporting requirements for compressors?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Project Structure

```
oilgasai/
├── inference/
│   ├── load_model.py        # Model loader (local 4-bit or HF API)
│   ├── chat.py              # Single/multi-turn inference
│   └── batch_inference.py   # Bulk compliance queries
├── rag/
│   ├── embeddings.py        # BGE-large-en-v1.5 embeddings
│   ├── faiss_store.py       # FAISS vector store (37k vectors)
│   └── rag_pipeline.py      # Full RAG chain
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/
│       ├── chat.py          # /chat endpoint
│       ├── compliance.py    # /compliance/report endpoint
│       └── sensors.py       # /sensors/calibrate, /sensors/ldar
├── sensors/
│   ├── calibration.py       # MIRA/Figaro/Alphasense drift correction
│   └── ldar.py              # LDAR analysis with NSPS thresholds
├── scripts/
│   ├── build_faiss_index.py # Build RAG vector index from docs
│   └── test_model.py        # Smoke test
├── tests/
│   ├── test_inference.py
│   ├── test_sensors.py
│   └── test_api.py
├── docs/API.md
├── .github/workflows/ci.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | General domain Q&A |
| POST | `/compliance/report` | Auto-generate compliance report |
| POST | `/compliance/check` | Targeted regulatory question |
| POST | `/sensors/calibrate` | Calibrate low-cost sensor readings |
| POST | `/sensors/ldar` | Analyze LDAR survey data |
| GET | `/health` | Health check |

Full interactive docs at `/docs` (Swagger UI) when the server is running.

---

## Environmental Impact

| Metric | Value |
|---|---|
| GPU-hours consumed | ~200 of 80,000 allocated |
| Hardware | 4× NVIDIA A100 64GB SXM4 per node |
| Location | CINECA, Bologna, Italy |
| Estimated CO₂ | ~15 kg CO₂eq (Italy grid: ~0.35 kgCO₂/kWh) |

---

## Funding & Support

- **UNDP AI Hub for Sustainable Development** — Compute Accelerator Programme (10,000 GPU hours)
- **CINECA ISCRA Programme** — 80,000 GPU hours on Leonardo HPC
- **Colorado State University** — METEC / SABER Project (methane sensor research foundation)
- **Meta AI** — Llama 3.1 open-source model family

---

## ⚠️ Disclaimer

OilGasAI Model Alpha is an expert assistant, not an autonomous decision-maker. All regulatory submissions, safety decisions, and engineering calculations must be verified by qualified professionals. Regulations change frequently — always verify against current EPA, state, and international requirements before submission.

---

## License

[Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)

---

## Contact

**Elijah Kiplimo** — Founder, OilGasAI  
🌐 [OilGasAI Platform](https://innovations1-spec.github.io/oilgasai/)  
🤗 [Hugging Face](https://huggingface.co/OilgasAI)
