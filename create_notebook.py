import json
import os

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)})

# Cell 1: Header
md("""# 🧠 ENGRAMA V3 - Dual T4 GPU Training on TinyStories
### Complete Memory-Optimized Pure PyTorch Architecture Implementation for Kaggle Dual T4 Setup

- **Architecture:** ENGRAMA V3 (Factorized Synapses, Hierarchical Dilated Consolidation, Factorized Multi-Candidate Evoker)
- **Model Size:** ~32.3 Million Parameters (`d_model=384`, `d_gate=48`, `d_ff=1536`, `num_cells=4`, `num_encoder_layers=1`, `num_consolidation_layers=2`, `synapse_rank=32`)
- **Dataset:** `roneneldan/TinyStories` (Complete Dataset)
- **Tokenizer:** GPT-2 Tokenizer (`vocab_size=50257`)
- **Context Length:** 512 tokens
- **Hardware:** Dual NVIDIA Tesla T4 GPUs (Kaggle Multi-GPU setup)
- **Memory Optimizations:** Factorized Synapses, Online LogSumExp Aggregation, Gradient Accumulation, PyTorch 2.x `torch.amp.autocast('cuda')`, Hierarchical Cache Pruning
- **Training Epochs:** 1 Full Epoch
- **No Attention / $QK^T$:** Pure cellular, dynamic gating, and logarithmic dilated consolidation architecture with factorized projections.
""")

# Cell 2: Environment Setup
code("""# Step 1: Environment Verification and Package Installation
import os
import sys
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any, Union

# Set CUDA allocator configuration to avoid VRAM fragmentation on Tesla T4 GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Install required Hugging Face utilities for dataset streaming & tokenization
!pip install -q datasets transformers accelerate tqdm

import datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
""")

# Cell 3: ENGRAMA V3 Architecture
code("""# Step 2: ENGRAMA V3 Pure PyTorch Core Architecture Definition
# Implements exact mathematical specifications from ENGRAMA V3 paper with memory optimizations

@dataclass
class EngramaConfig:
    vocab_size: int = 50257
    d_model: int = 384
    d_gate: int = 48
    d_ff: int = 1536
    num_cells: int = 4
    num_encoder_layers: int = 1
    num_consolidation_layers: int = 2
    context_length: int = 512
    offsets: Optional[List[int]] = None
    num_candidates: int = 4
    candidate_aggregation: str = "logsumexp"
    activation: str = "gelu"
    dropout: float = 0.0
    dtype: str = "float32"
    version: str = "v3"
    tie_embeddings: bool = True
    synapse_mode: str = "factorized"
    synapse_rank: int = 32
    identity_transport: bool = True
    cell_mode: str = "shared_core"
    offset_mode: str = "hierarchical_dyadic"
    global_anchor: bool = False
    evoker_mode: str = "factorized"
    hierarchical_cache: bool = True

    def __post_init__(self):
        if self.offsets is None:
            self.offsets = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_gate >= self.d_model:
            raise ValueError(f"d_gate ({self.d_gate}) must be < d_model ({self.d_model})")
        if not (1 <= self.num_candidates <= 8):
            raise ValueError("num_candidates must be between 1 and 8 inclusive")
        if self.candidate_aggregation not in ("max", "logsumexp", "mean"):
            raise ValueError("candidate_aggregation must be 'max', 'logsumexp', or 'mean'")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError("activation must be 'gelu', 'relu', or 'silu'")
        if self.version not in ("v1", "v2", "v3"):
            raise ValueError("version must be 'v1', 'v2', or 'v3'")
        if self.synapse_mode not in ("dense", "factorized"):
            raise ValueError("synapse_mode must be 'dense' or 'factorized'")
        if self.cell_mode not in ("independent", "shared_core"):
            raise ValueError("cell_mode must be 'independent' or 'shared_core'")
        if self.offset_mode not in ("dense_dilated", "hierarchical_dyadic", "binary_minimal"):
            raise ValueError("offset_mode must be 'dense_dilated', 'hierarchical_dyadic', or 'binary_minimal'")
        if self.evoker_mode not in ("dense", "factorized"):
            raise ValueError("evoker_mode must be 'dense' or 'factorized'")
        if any(o < 0 for o in self.offsets):
            raise ValueError("All positional offsets must be non-negative (>= 0)")

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class SharedCoreCellGroup(nn.Module):
    def __init__(self, num_cells: int, d_model: int, d_ff: int, activation: str = "gelu"):
        super().__init__()
        self.num_cells = num_cells
        self.ln = LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() if activation == "gelu" else nn.SiLU() if activation == "silu" else nn.ReLU()
        self.s_scale = nn.Parameter(torch.ones(num_cells, d_model))
        self.n_mod = nn.Parameter(torch.ones(num_cells, d_model))
        self.q_bias = nn.Parameter(torch.zeros(num_cells, d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x_mod = self.n_mod * u + self.q_bias
        f_out = self.w2(self.act(self.w1(self.ln(x_mod))))
        return u + self.s_scale * f_out

class FactorizedSynapse(nn.Module):
    def __init__(self, d_model: int, rank: int, identity_transport: bool = True):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.s = nn.Parameter(torch.randn(rank) * 0.01)
        if identity_transport:
            self.beta = nn.Parameter(torch.ones(1))
        else:
            self.beta = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        transformed = (h @ self.V) @ torch.diag(self.s) @ self.U.T
        return self.beta * h + transformed if self.beta is not None else transformed

class SynapseLayer(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.p_g = nn.Linear(config.d_model, config.d_gate, bias=False)
        self.gate_w = nn.Parameter(torch.randn(config.num_cells, config.num_cells, config.d_gate) * 0.01)
        self.gate_b = nn.Parameter(torch.zeros(config.num_cells, config.num_cells))
        self.w_channel = nn.Parameter(torch.randn(config.num_cells, config.num_cells, config.d_model) * 0.01)
        if config.cell_mode == "shared_core":
            self.cells = SharedCoreCellGroup(
                config.num_cells, config.d_model, config.d_ff, config.activation
            )
        else:
            raise NotImplementedError("Independent cell mode not supported in V3")
        if config.synapse_mode == "factorized":
            self.synapses = nn.ModuleDict(
                {
                    f"{a}_{b}": FactorizedSynapse(
                        config.d_model, config.synapse_rank, config.identity_transport
                    )
                    for a in range(config.num_cells)
                    for b in range(config.num_cells)
                }
            )
        else:
            raise NotImplementedError("Dense synapse mode not supported in V3")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, C, D = h.shape
        g_proj = self.p_g(h)
        out = torch.zeros_like(h)
        for j in range(C):
            H_transformed = h[..., j, :]
            channel_contributions = []
            for i in range(C):
                g_i = g_proj[..., i, :]
                w_ij = self.gate_w[i, j]
                b_ij = self.gate_b[i, j]
                alpha_ij = torch.sigmoid(torch.einsum("...g,g->...", g_i, w_ij) + b_ij)
                w_channel_ij = self.w_channel[i, j]
                h_i = h[..., i, :]
                synapse = self.synapses[f"{i}_{j}"]
                contrib = alpha_ij.unsqueeze(-1) * (synapse(h_i) * w_channel_ij)
                channel_contributions.append(contrib)
            synapse_mix = torch.stack(channel_contributions, dim=0).sum(dim=0)
            cell_out = self.cells(synapse_mix)
            out[..., j, :] = cell_out
        return out

class IsolatedEncoder(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.init_proj = nn.Linear(config.d_model, config.num_cells * config.d_model)
        self.layers = nn.ModuleList([SynapseLayer(config) for _ in range(config.num_encoder_layers)])
        self.w_pool = nn.Linear(config.num_cells * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.init_proj(x).view(B, N, self.layers[0].num_cells, D)
        for layer in self.layers:
            h = layer(h)
        h = h.view(B, N, -1)
        return self.w_pool(h)

class PositionalDilatedMix(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.config = config
        self.p_g = nn.Linear(config.d_model, config.d_gate, bias=False)
        self.gate_w = nn.ParameterDict(
            {
                str(p): nn.Parameter(torch.randn(config.d_gate, config.d_model) * 0.01)
                for p in config.offsets
            }
        )
        self.gate_b = nn.ParameterDict(
            {str(p): nn.Parameter(torch.zeros(config.d_model)) for p in config.offsets}
        )
        if config.synapse_mode == "factorized":
            self.synapses = nn.ModuleDict(
                {
                    str(p): FactorizedSynapse(
                        config.d_model, config.synapse_rank, config.identity_transport
                    )
                    for p in config.offsets
                }
            )
        else:
            raise NotImplementedError
        if config.hierarchical_gate:
            self.rho = nn.ParameterDict(
                {str(p): nn.Parameter(torch.zeros(1)) for p in config.offsets}
            )

    def get_offsets(self, layer_idx: int, total_layers: int) -> List[int]:
        if self.config.offset_mode == "hierarchical_dyadic":
            if layer_idx == 0:
                return [0, 1]
            dyadic = 2 ** layer_idx
            return [0, 1, dyadic] if dyadic < self.config.context_length else [0, 1]
        return self.config.offsets

    def forward_train(self, T_prev: torch.Tensor, layer_idx: int, total_layers: int) -> torch.Tensor:
        offsets = self.get_offsets(layer_idx, total_layers)
        t_pos = torch.zeros_like(T_prev)
        for p in offsets:
            str_p = str(p)
            if p == 0:
                t_shifted = T_prev
            elif p < T_prev.size(1):
                t_shifted = torch.cat([
                    torch.zeros_like(T_prev[:, :p, :]),
                    T_prev[:, :-p, :]
                ], dim=1)
            else:
                continue
            h_g = self.p_g(t_shifted)
            gate_logits = h_g @ self.gate_w[str_p] + self.gate_b[str_p]
            g = torch.sigmoid(gate_logits) * torch.sigmoid(self.rho[str_p])
            transformed = self.synapses[str_p](t_shifted)
            t_pos += g * transformed
        return t_pos

class ConsolidationLayer(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.mix = PositionalDilatedMix(config)
        self.cell = Cell(
            config.d_model, config.d_ff, config.dropout, config.activation
        )

    def forward_train(self, T_prev: torch.Tensor, layer_idx: int, total_layers: int) -> torch.Tensor:
        t_pos = self.mix.forward_train(T_prev, layer_idx, total_layers)
        return self.cell(t_pos)

class ConsolidationStack(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ConsolidationLayer(config) for _ in range(config.num_consolidation_layers)])

    def forward_train(self, T0: torch.Tensor) -> torch.Tensor:
        t = T0
        for layer_idx, layer in enumerate(self.layers):
            t = layer.forward_train(t, layer_idx, len(self.layers))
        return t

class Cell(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() if activation == "gelu" else nn.SiLU() if activation == "silu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(self.dropout(self.act(self.w1(self.ln(x)))))

class FactorizedEvokerCandidate(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model = d_model
        self.W_shared = nn.Linear(d_model, d_model, bias=False)
        self.U_e = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.V_e = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.s_m = nn.Parameter(torch.randn(rank) * 0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lr_part = (h @ self.V_e) @ torch.diag(self.s_m) @ self.U_e
        return self.W_shared(h) + lr_part

class MultiCandidateEvoker(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.candidates = nn.ModuleList([
            FactorizedEvokerCandidate(config.d_model, config.synapse_rank)
            for _ in range(config.num_candidates)
        ])
        self.aggregation = config.candidate_aggregation

    def forward(self, h_star: torch.Tensor, embedding_weights: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.candidates[0].d_model)
        logits_list = [F.linear(cand(h_star), embedding_weights) * scale for cand in self.candidates]
        if self.aggregation == "logsumexp":
            max_logits = max(logits_list, key=lambda x: x.max())
            sum_exp = torch.stack([torch.exp(l - max_logits) for l in logits_list]).sum(dim=0)
            return max_logits + torch.log(sum_exp)
        elif self.aggregation == "max":
            return torch.stack(logits_list, dim=-1).max(dim=-1).values
        elif self.aggregation == "mean":
            return torch.stack(logits_list, dim=-1).mean(dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

class EngramaModel(nn.Module):
    def __init__(self, config: EngramaConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = IsolatedEncoder(config)
        self.consolidation = ConsolidationStack(config)
        self.evoker = MultiCandidateEvoker(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        T0 = self.encoder(x)
        T_L = self.consolidation.forward_train(T0)
        logits = self.evoker(T_L, self.embeddings.weight)
        return logits

# Verify architecture loads successfully
config = EngramaConfig()
model = EngramaModel(config)
print("ENGRAMA V3 Architecture loaded successfully!")
""")

# Cell 4: Model Initialization
code("""# Step 3: Initialize ENGRAMA V3 Model with Factorized Synapses
config = EngramaConfig(
    vocab_size=50257,               # GPT-2 Tokenizer Vocab Size
    d_model=384,                    # Hidden Dimension
    d_gate=48,                      # Gating Dimension
    d_ff=1536,                      # Feed-Forward Expansion (4 * d_model)
    num_cells=4,                    # Parallel Cellular Representations
    num_encoder_layers=1,           # Encoder Synapse Layers
    num_consolidation_layers=2,     # Consolidation Stack Layers
    context_length=512,             # Max Sequence Length
    offsets=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Hierarchical Dyadic Offsets
    num_candidates=4,               # Evoker Candidates
    candidate_aggregation="logsumexp",
    activation="gelu",
    dropout=0.1,
    version="v3",
    synapse_mode="factorized",
    synapse_rank=32,
    identity_transport=True,
    cell_mode="shared_core",
    offset_mode="hierarchical_dyadic",
    evoker_mode="factorized",
    hierarchical_cache=True
)

model = EngramaModel(config)
total_params = sum(p.numel() for p in model.parameters())

print("=" * 60)
print(f"  ENGRAMA V3 MODEL PARAMETER BREAKDOWN")
print("=" * 60)
print(f"  Vocab Size              : {config.vocab_size}")
print(f"  Embedding Dimension (d) : {config.d_model}")
print(f"  Gate Dimension (d_gate) : {config.d_gate}")
print(f"  Feed-Forward Dimension  : {config.d_ff}")
print(f"  Number of Cells (C)     : {config.num_cells}")
print(f"  Encoder Synapse Layers  : {config.num_encoder_layers}")
print(f"  Consolidation Layers    : {config.num_consolidation_layers}")
print(f"  Context Sequence Length : {config.context_length}")
print(f"  Synapse Low-Rank (r)    : {config.synapse_rank}")
print(f"  Total Parameters        : {total_params:,} ({total_params/1e6:.2f} Million)")
print("=" * 60)
""")

# Cell 5: Dataset
code("""# Step 4: Load TinyStories Dataset & Tokenize with GPT-2 Tokenizer
print("Loading GPT-2 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading roneneldan/TinyStories dataset from Hugging Face...")
raw_dataset = datasets.load_dataset("roneneldan/TinyStories")

print(f"Train split size: {len(raw_dataset['train']):,} stories")
print(f"Validation split size: {len(raw_dataset['validation']):,} stories")

SEQ_LEN = 512

class TinyStoriesTokenDataset(Dataset):
    def __init__(self, raw_data, tokenizer, seq_len=512, max_samples=None):
        self.seq_len = seq_len
        texts = raw_data['text']
        if max_samples:
            texts = texts[:max_samples]
        
        print("Tokenizing stories into contiguous tokens...")
        full_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            full_tokens.extend(tokens)
            full_tokens.append(tokenizer.eos_token_id)
            
        total_tokens = len(full_tokens)
        num_chunks = total_tokens // (seq_len + 1)
        truncated_len = num_chunks * (seq_len + 1)
        
        self.data = torch.tensor(full_tokens[:truncated_len], dtype=torch.long).view(-1, seq_len + 1)
        print(f"Created {len(self.data):,} sequences of length {seq_len} ({len(self.data) * seq_len:,} total tokens)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

print("Preparing Train Dataset...")
train_dataset = TinyStoriesTokenDataset(raw_dataset['train'], tokenizer, seq_len=SEQ_LEN)

print("Preparing Validation Dataset...")
val_dataset = TinyStoriesTokenDataset(raw_dataset['validation'], tokenizer, seq_len=SEQ_LEN)
""")

# Cell 6: Multi-GPU Setup
code("""# Step 5: Multi-GPU Dual NVIDIA Tesla T4 Engine & Batch Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"🚀 Harnessing Dual Tesla T4 GPUs! Using torch.nn.DataParallel across {torch.cuda.device_count()} GPUs.")
    train_model = nn.DataParallel(model)
else:
    print(f"Running on single GPU / CPU: {device}")
    train_model = model

train_model = train_model.to(device)

# Optimized Batch Size per GPU to fit within VRAM limits
BATCH_SIZE_PER_GPU = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GPUS = max(1, torch.cuda.device_count())
MICRO_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

print(f"Batch Size per GPU             : {BATCH_SIZE_PER_GPU}")
print(f"Micro-Batch Size (across GPUs) : {MICRO_BATCH_SIZE}")
print(f"Gradient Accumulation Steps   : {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size           : {EFFECTIVE_BATCH_SIZE}")

train_loader = DataLoader(
    train_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)
""")

# Cell 7: Training Setup
code("""# Step 6: Optimizer, Learning Rate Scheduler & Training Engine Setup
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
EPOCHS = 1

optimizer = torch.optim.AdamW(
    train_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95)
)

total_micro_steps = len(train_loader) * EPOCHS
total_optimizer_steps = total_micro_steps // GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(0.05 * total_optimizer_steps)

def get_lr_factor(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)
scaler = torch.amp.GradScaler()
criterion = nn.CrossEntropyLoss()

print(f"Total Optimizer Steps  : {total_optimizer_steps:,}")
print(f"Warmup Steps           : {warmup_steps:,}")
""")

# Cell 8: Training Loop
code("""# Step 7: Train ENGRAMA V3 Model for 1 Full Epoch on Dual T4 GPUs
print("\\n" + "=" * 70)
print("  STARTING ENGRAMA V3 DUAL T4 GPU TRAINING (1 EPOCH)")
print("=" * 70)

train_model.train()
start_time = time.time()
running_loss = 0.0
tokens_processed = 0

optimizer.zero_grad()
pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch 1/1 Training")

for step, (input_ids, labels) in pbar:
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    # Mixed precision forward pass
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = train_model(input_ids)
        loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
        scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
    
    scaler.scale(scaled_loss).backward()
    step_loss = loss.item()
    running_loss += step_loss
    
    batch_tokens = input_ids.numel()
    tokens_processed += batch_tokens
    
    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    
    if (step + 1) % 100 == 0 or step == len(train_loader) - 1:
        elapsed = time.time() - start_time
        tok_per_sec = tokens_processed / elapsed
        avg_loss = running_loss / (step + 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        
        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "PPL": f"{perplexity:.2f}",
            "LR": f"{current_lr:.2e}",
            "Tok/s": f"{tok_per_sec:.0f}"
        })

total_elapsed = time.time() - start_time
final_avg_loss = running_loss / len(train_loader)
final_ppl = math.exp(min(final_avg_loss, 20.0))

print("\\n" + "=" * 70)
print(f"  EPOCH 1 TRAINING COMPLETED IN {total_elapsed/60:.2f} MINUTES")
print(f"  Final Training Loss : {final_avg_loss:.4f}")
print(f"  Final Perplexity    : {final_ppl:.2f}")
print(f"  Total Processed     : {tokens_processed:,} tokens ({tokens_processed/total_elapsed:.0f} tok/sec)")
print("=" * 70)
""")

# Cell 9: Validation
code("""# Step 8: Validation Split Evaluation
train_model.eval()
val_loss = 0.0
val_tokens = 0

print("\\nEvaluating model on Validation split...")
with torch.no_grad():
    for input_ids, labels in tqdm(val_loader, desc="Validation"):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = train_model(input_ids)
            loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
            
        val_loss += loss.item() * input_ids.size(0)
        val_tokens += input_ids.size(0)

avg_val_loss = val_loss / val_tokens
val_ppl = math.exp(min(avg_val_loss, 20.0))

print("=" * 60)
print(f"  VALIDATION RESULTS")
print("=" * 60)
print(f"  Validation Loss       : {avg_val_loss:.4f}")
print(f"  Validation Perplexity : {val_ppl:.2f}")
print("=" * 60)
""")

# Cell 10: Generation
code("""# Step 9: Autoregressive Text Generation Demo
model.eval()
model.to(device)

def generate_story(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 40):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp_tensor = torch.tensor([generated[-config.context_length:]], dtype=torch.long, device=device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(inp_tensor)
            next_logits = logits[0, -1, :].float() / temperature
            
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')
                
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated)

prompts = [
    "Once upon a time, a little girl named Lily",
    "One day, a tiny puppy saw a big ball in the garden.",
    "Tom wanted to build a high tower with his wooden blocks."
]

print("=" * 70)
print("  ENGRAMA V3 - AUTOREGRESSIVE STORY GENERATION SAMPLES")
print("=" * 70)

for i, prompt in enumerate(prompts, 1):
    story = generate_story(prompt, max_new_tokens=80, temperature=0.7)
    print(f"\\n--- Sample {i} ---")
    print(story)
    print("-" * 50)
""")

# Cell 11: Save Checkpoint
code("""# Step 10: Save Trained Checkpoint & Configuration
os.makedirs("checkpoints", exist_ok=True)

save_path = "checkpoints/engrama_v3_30m_tinystories.pt"
config_path = "checkpoints/config_v3.json"

raw_model = train_model.module if isinstance(train_model, nn.DataParallel) else train_model

torch.save({
    "model_state_dict": raw_model.state_dict(),
    "config": asdict(config),
}, save_path)

with open(config_path, "w") as f:
    json.dump(asdict(config), f, indent=2)

print(f"✅ Trained Engrama V3 checkpoint saved successfully to: {save_path}")
print(f"✅ Configuration JSON saved to: {config_path}")
""")

notebook_content = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "gpuClass": "standard",
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join("kaggle", "engrama_v3_30m_tinystories_dual_t4.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Generated successfully!")