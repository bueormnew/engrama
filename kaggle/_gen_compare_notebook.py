#!/usr/bin/env python3
"""Generate kaggle/engrama_v4_vs_ablation_transformer_2xt4.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKER = (ROOT / "train_compare_ddp.py").read_text(encoding="utf-8")
OUT = ROOT / "engrama_v4_vs_ablation_transformer_2xt4.ipynb"


def md(src: str) -> dict:
    if not src.endswith("\n"):
        src += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    if not src.endswith("\n"):
        src += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(md("""# ENGRAMA V4 vs ablaciones vs Transformer — 2×T4, GPT-2, 100M tokens

Comparación **controlada** de cuatro modelos de ~20M parámetros, entrenados con el **mismo** tokenizer GPT-2, el **mismo** corpus (~100M tokens), las **mismas** épocas y el **mismo** recetario de entrenamiento (DDP + AMP fp16 + CE lineal fusionada).

| Modelo | Qué es | Qué se le quita |
|---|---|---|
| `engrama_v4` | ENGRAMA V4 **completo** | nada (dual gating + Trace Tap T0 + offsets resonantes + RMSNorm + latent fusion) |
| `engrama_source_gate` | ENGRAMA V4 limitado | **gating dual target-source** → gating V3 source-only |
| `engrama_no_tracetap` | ENGRAMA V4 limitado | **Trace Tap T0** (bypass a la huella prístina) |
| `transformer` | Decoder GPT con RoPE + RMSNorm + SDPA causal | — (baseline con atención \(O(N^2)\)) |

**Hardware objetivo:** Kaggle 2× Tesla T4 16 GB. **Presupuesto:** < 3 h para los 4 entrenamientos + mediciones.

- Tokenizer: **GPT-2 BPE** (50,257)
- Contexto de entrenamiento: **512**
- Tokens de train: **100,000,000** (1 época)
- Batch: **16 / GPU** (global 32 en 2×T4)
- Optimizador: AdamW fused `lr=3e-4`, `betas=(0.9, 0.95)`, warmup 500 + cosine, clip 1.0
- Pérdida: proyección + CE **por chunks de posiciones** (nunca se materializa `(B,N,50k)` completo; CE en fp32)
- Multi-GPU: **DDP/NCCL** (no `DataParallel`)

Al final: pérdida, perplejidad, tokens/s, VRAM vs contexto, orden de complejidad, recuperación KV de largo alcance, y una tabla única de comparación.

> Autor: BUEORM · Licencia AGPL-3.0 · Arquitectura ENGRAMA intacta (solo se optimiza el runtime).
"""))

cells.append(md("## 0. Instalación y flags CUDA (anti-NaN)"))

cells.append(code(r"""import os, sys, math, time, json, gc, glob, shutil, subprocess, random, traceback
from pathlib import Path

# Prefer a local checkout (this repo) over PyPI; fall back to GitHub.
def _install():
    cands = [
        Path.cwd(),
        Path.cwd().parent,
        Path('/kaggle/working/engrama'),
        Path('/kaggle/input/engrama'),
    ]
    for p in cands:
        if (p / 'src' / 'engrama').is_dir() and (p / 'pyproject.toml').is_file():
            print('Instalando ENGRAMA editable desde', p)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(p),
                                   'transformers', 'numpy'])
            return
    print('Instalando ENGRAMA desde GitHub')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'git+https://github.com/bueormnew/engrama.git',
                           'transformers', 'numpy'])

_install()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import engrama
from engrama import EngramaConfig, EngramaModel, Generator, linear_cross_entropy

print('ENGRAMA', engrama.__version__, '| torch', torch.__version__)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NGPU = torch.cuda.device_count() if DEVICE == 'cuda' else 0
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        # GEMM fp16 con reduccion fp16 desborda el residual V4 a Inf/NaN tras el warmup.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)  # T4 = SM75, sin FlashAttention
    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(NGPU)])
    for i in range(NGPU):
        print('  cuda:%d  %.1f GiB' % (i, torch.cuda.get_device_properties(i).total_memory / 2**30))
else:
    print('SIN GPU — activa 2x T4 en Kaggle. FAST_MODE permite un humo en CPU.')
"""))

cells.append(md("## 1. Configuración central (un solo sitio)"))

cells.append(code(r"""FAST_MODE = False   # True = humo (pocos pasos). False = experimento real < 3 h en 2x T4.
SEED = 1234
ARCHS = ['engrama_v4', 'engrama_source_gate', 'engrama_no_tracetap', 'transformer']

SEQ_LEN = 512
VOCAB_SIZE = 50257
TARGET_TRAIN_TOKENS = 100_000_000
TARGET_VALID_TOKENS = 2_000_000

# Identicos para los 4 modelos
EPOCHS = 1
LOCAL_BATCH = 16          # por GPU; global = LOCAL_BATCH * nproc
EVAL_BATCH = 8
LR = 3e-4                 # 6e-4 + AMP fp16 reventaba a NaN ~paso 350
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 500
LOG_EVERY = 50
EVAL_EVERY = 500
EVAL_BATCHES = 25
LINEAR_CHUNK = 2048
COMPILE = True
COMPILE_MODE = 'default'  # max-autotune gasta demasiado compile-time x 4 modelos
RESUME = True

TRAIN_FILE = 'tinystories_train.txt'
VALID_FILE = 'tinystories_valid.txt'
TRAIN_IDS = 'tinystories_train.ids'
VALID_IDS = 'tinystories_valid.ids'
TRAIN_BYTES = 2_227_753_162
VALID_BYTES = 22_502_601

WORK_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('./compare_working')
SAVE_ROOT = os.path.join(WORK_DIR, 'compare_ckpts')
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)

NPROC = max(1, min(2, NGPU)) if NGPU else 1
GLOBAL_BATCH = LOCAL_BATCH * NPROC
TOKENS_PER_STEP = GLOBAL_BATCH * SEQ_LEN
# 100M / (32*512) ≈ 6104 pasos. A 0.25 s/paso x 4 modelos ≈ 1.7 h de train.
EST_STEPS = TARGET_TRAIN_TOKENS // TOKENS_PER_STEP

if FAST_MODE:
    TARGET_TRAIN_TOKENS = 50_000
    TARGET_VALID_TOKENS = 10_000
    LOCAL_BATCH = 4
    WARMUP_STEPS = 5
    LOG_EVERY = 2
    EVAL_EVERY = 8
    EVAL_BATCHES = 2
    COMPILE = False
    EPOCHS = 1
    NPROC = 1
    GLOBAL_BATCH = LOCAL_BATCH * NPROC
    TOKENS_PER_STEP = GLOBAL_BATCH * SEQ_LEN
    EST_STEPS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('modo=%s  GPUs=%d nproc=%d  seq=%d  global_batch=%d  ~pasos/modelo=%d' % (
    'FAST' if FAST_MODE else 'FULL', NGPU, NPROC, SEQ_LEN, GLOBAL_BATCH, EST_STEPS))
print('tokens train objetivo=%s  tokens/paso=%s  presupuesto train ≈ %.1f h (4 modelos, 0.28 s/paso)' % (
    format(TARGET_TRAIN_TOKENS, ','), format(TOKENS_PER_STEP, ','),
    (EST_STEPS * 0.28 * 4) / 3600.0))
print('ckpts ->', SAVE_ROOT)
if not FAST_MODE and NGPU == 0:
    raise RuntimeError('El modo FULL necesita GPU (idealmente 2x T4 en Kaggle).')
if not FAST_MODE and NGPU < 2:
    print('AVISO: se esperaban 2x T4. Se entrenara con nproc=%d (mismo recetario, menos tok/s).' % NPROC)
"""))

cells.append(md("## 2. TinyStories (descarga robusta) + tokenizer GPT-2"))

cells.append(code(r"""def iter_stories(path):
    buf = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                buf.append(line.rstrip('\n'))
            elif buf:
                yield ' '.join(buf).strip()
                buf = []
    if buf:
        yield ' '.join(buf).strip()


def download_verified(url, path, expected_bytes, retries=8):
    import urllib.request
    dest = path if os.path.isabs(path) else os.path.join(WORK_DIR, path)
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    if done == expected_bytes:
        print('  %s: ya completo (%.0f MB)' % (dest, done / 2**20))
        return dest
    if done > expected_bytes:
        os.remove(dest)
        done = 0
    for attempt in range(1, retries + 1):
        mode = 'ab' if done else 'wb'
        headers = {'Range': 'bytes=%d-' % done} if done else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = done if getattr(resp, 'status', 200) == 206 else 0
                if getattr(resp, 'status', 200) == 200:
                    mode, total = 'wb', 0
                os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                        if total % (256 * 1024 * 1024) < 4 * 1024 * 1024:
                            print('    %s: %.0f MB ...' % (os.path.basename(dest), total / 2**20))
            done = os.path.getsize(dest)
            if done == expected_bytes:
                print('  %s: OK (%.0f MB)' % (dest, done / 2**20))
                return dest
            print('  incompleto (%d != %d); reintento ...' % (done, expected_bytes))
        except Exception as exc:
            done = os.path.getsize(dest) if os.path.exists(dest) else 0
            print('  %s: %s; reintento %d/%d' % (os.path.basename(dest), type(exc).__name__, attempt, retries))
            time.sleep(min(30.0, 2 ** attempt))
            if attempt >= retries:
                raise
    raise RuntimeError('Descarga fallida: ' + url)


def find_kaggle_file(basename, expected):
    for pat in ('/kaggle/input/*/' + basename, '/kaggle/input/**/' + basename):
        for hit in sorted(glob.glob(pat, recursive=True)):
            if os.path.getsize(hit) == expected:
                return hit
    return None

TRAIN_URL = ('https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/'
             'TinyStoriesV2-GPT4-train.txt')
VALID_URL = ('https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/'
             'TinyStoriesV2-GPT4-valid.txt')

print('Localizando TinyStories ...')
train_path = find_kaggle_file('TinyStoriesV2-GPT4-train.txt', TRAIN_BYTES)
valid_path = find_kaggle_file('TinyStoriesV2-GPT4-valid.txt', VALID_BYTES)
if FAST_MODE:
    if valid_path is None:
        valid_path = download_verified(VALID_URL, VALID_FILE, VALID_BYTES)
    train_path = valid_path
else:
    if train_path is None:
        train_path = download_verified(TRAIN_URL, TRAIN_FILE, TRAIN_BYTES)
    else:
        print('  train montado:', train_path)
    if valid_path is None:
        valid_path = download_verified(VALID_URL, VALID_FILE, VALID_BYTES)
    else:
        print('  valid montado:', valid_path)
print('train:', train_path)
print('valid:', valid_path)
"""))

cells.append(code(r'''class GPT2Adapter:
    """Tokenizer GPT-2 con la interfaz que espera engrama.Generator."""
    def __init__(self, hf_tok):
        self.tok = hf_tok
        eot = hf_tok.eos_token_id
        self.SPECIAL_TOKENS = {'<eos>': eot, '<bos>': eot, '<pad>': eot}
        self.vocab_size = len(hf_tok)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = list(self.tok.encode(text, add_special_tokens=False))
        if add_bos:
            ids = [self.SPECIAL_TOKENS['<bos>']] + ids
        if add_eos:
            ids = ids + [self.SPECIAL_TOKENS['<eos>']]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts):
        outs = self.tok(list(texts), add_special_tokens=False)['input_ids']
        eos = self.SPECIAL_TOKENS['<eos>']
        return [list(ids) + [eos] for ids in outs]


from transformers import GPT2TokenizerFast
_hf = GPT2TokenizerFast.from_pretrained('gpt2')
if len(_hf) != 50257:
    raise RuntimeError('GPT-2 vocab inesperado: %d' % len(_hf))
tokenizer = GPT2Adapter(_hf)
assert tokenizer.vocab_size == VOCAB_SIZE
EOS_ID = tokenizer.SPECIAL_TOKENS['<eos>']
print('Tokenizer GPT-2 BPE  vocab =', tokenizer.vocab_size, ' eos =', EOS_ID)
'''))

cells.append(md("## 3. Tokenización streaming → memmap int32 (corte exacto a 100M)"))

cells.append(code(r"""class MemmapTokenWriter:
    def __init__(self, out_raw, initial_capacity):
        self.path = out_raw
        self.cap = max(1024, int(initial_capacity))
        self.mm = np.memmap(out_raw, dtype=np.int32, mode='w+', shape=(self.cap,))
        self.pos = 0

    def ensure(self, extra):
        if self.pos + extra <= self.cap:
            return
        new_cap = max(self.cap * 2, self.pos + extra)
        self.mm.flush(); del self.mm
        with open(self.path, 'r+b') as f:
            f.truncate(new_cap * 4)
        self.mm = np.memmap(self.path, dtype=np.int32, mode='r+', shape=(new_cap,))
        self.cap = new_cap

    def extend(self, ids):
        self.ensure(len(ids))
        self.mm[self.pos:self.pos + len(ids)] = np.asarray(ids, dtype=np.int32)
        self.pos += len(ids)

    def finalize(self, max_ids=None):
        n = int(self.pos if max_ids is None else min(self.pos, max_ids))
        window = SEQ_LEN + 1
        n = (n // window) * window
        self.mm.flush(); del self.mm
        with open(self.path, 'r+b') as f:
            f.truncate(n * 4)
        return np.memmap(self.path, dtype=np.int32, mode='r', shape=(n,))


def stories_to_memmap(path, out_raw, tokenizer, max_ids, batch_stories=256):
    if os.path.exists(out_raw) and os.path.getsize(out_raw) >= 4 * max_ids:
        n = os.path.getsize(out_raw) // 4
        window = SEQ_LEN + 1
        n = min(n, max_ids)
        n = (n // window) * window
        mm = np.memmap(out_raw, dtype=np.int32, mode='r', shape=(n,))
        print('  reusando %s (%s tokens)' % (out_raw, format(n, ',')))
        return mm
    writer = MemmapTokenWriter(out_raw, initial_capacity=max_ids + 4096)
    batch, n_stories = [], 0
    for story in iter_stories(path):
        batch.append(story)
        if len(batch) < batch_stories:
            continue
        for ids in tokenizer.encode_batch(batch):
            if ids:
                writer.extend(ids)
        n_stories += len(batch)
        batch = []
        if n_stories % 20000 < batch_stories:
            print('  %7d cuentos | %10d tokens ...' % (n_stories, writer.pos))
        if writer.pos >= max_ids:
            break
    if batch and writer.pos < max_ids:
        for ids in tokenizer.encode_batch(batch):
            if ids:
                writer.extend(ids)
        n_stories += len(batch)
    print('  %7d cuentos | %10d tokens (pre-corte)' % (n_stories, writer.pos))
    mm = writer.finalize(max_ids=max_ids)
    print('  corte final: %s tokens (%s ventanas x %d)' % (
        format(len(mm), ','), format(len(mm) // (SEQ_LEN + 1), ','), SEQ_LEN))
    return mm

train_ids_path = os.path.join(WORK_DIR, TRAIN_IDS)
valid_ids_path = os.path.join(WORK_DIR, VALID_IDS)
t0 = time.time()
print('Tokenizando train hasta', format(TARGET_TRAIN_TOKENS, ','), 'tokens ...')
train_mm = stories_to_memmap(train_path, train_ids_path, tokenizer, TARGET_TRAIN_TOKENS)
print('Tokenizando valid hasta', format(TARGET_VALID_TOKENS, ','), 'tokens ...')
valid_mm = stories_to_memmap(valid_path, valid_ids_path, tokenizer, TARGET_VALID_TOKENS)
print('Tokenizacion en %ds | train=%s | valid=%s' % (
    int(time.time() - t0), format(len(train_mm), ','), format(len(valid_mm), ',')))
if not FAST_MODE and len(train_mm) < 90_000_000:
    raise RuntimeError('Train tokenizado demasiado corto: %d (se esperaban ~100M)' % len(train_mm))
"""))

cells.append(md("## 4. Worker DDP (se escribe a disco para `torchrun`)"))

cells.append(code("WORKER_SRC = " + repr(WORKER) + "\n" + r"""
worker_path = os.path.join(WORK_DIR, 'train_compare_ddp.py')
# Prefer the repo copy when present (keeps notebook and file in sync during development).
for c in (Path.cwd() / 'kaggle' / 'train_compare_ddp.py',
          Path.cwd() / 'train_compare_ddp.py',
          Path(WORK_DIR) / 'train_compare_ddp.py'):
    if c.is_file() and c.stat().st_size > 1000:
        shutil.copy2(c, worker_path)
        print('Worker copiado desde', c)
        break
else:
    with open(worker_path, 'w', encoding='utf-8') as f:
        f.write(WORKER_SRC)
    print('Worker escrito (bundle del notebook) ->', worker_path)

sys.path.insert(0, WORK_DIR)
import importlib
import train_compare_ddp as tcd
importlib.reload(tcd)
print('archs:', list(tcd.ARCH_SPECS))
"""))

cells.append(md("## 5. Verificar parámetros (~20M, diferencia < 10%)"))

cells.append(code(r"""cards = {}
for arch in ARCHS:
    m = tcd.build_raw_model(arch, VOCAB_SIZE, SEQ_LEN)
    card = tcd.model_card(arch, m)
    cards[arch] = card
    print('%s: %s params (%.2fM)  |  %s' % (
        arch, format(card['parameters'], ','), card['parameters'] / 1e6, card['title']))
    if arch.startswith('engrama'):
        rf = card['receptive_field']
        print('    gating=%s tap=%s offsets=%s  reach=%s covers=%s' % (
            card['gating_mode'], card['trace_tap'], card['offset_mode'],
            rf['max_reach'], rf['covers_context']))
    del m
    gc.collect()

nparams = [cards[a]['parameters'] for a in ARCHS]
lo, hi, mean = min(nparams), max(nparams), sum(nparams) / len(nparams)
print('\nrango %.2fM — %.2fM  (spread %.1f%% respecto a la media)' % (
    lo / 1e6, hi / 1e6, 100.0 * (hi - lo) / mean))
if lo < 10_000_000 or hi > 22_000_000:
    raise RuntimeError('Parametros fuera de 10-20M: %s' % nparams)
if (hi - lo) / mean > 0.12:
    raise RuntimeError('Los modelos no son comparables en tamaño (spread > 12%).')
print('OK: tamaños comparables.')
"""))

cells.append(md("""## 6. Humo (2–3 pasos) — si esto falla, no se lanza el entrenamiento de 3 h

Un forward+backward corto **por arquitectura** en 1 GPU, batch pequeño, sin `torch.compile`. Detecta OOM, shapes y NaNs inmediatos.
"""))

cells.append(code(r"""def smoke_arch(arch, steps=2):
    device = torch.device('cuda:0' if NGPU else 'cpu')
    model = tcd.build_raw_model(arch, VOCAB_SIZE, SEQ_LEN).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    B = 2 if device.type == 'cuda' else 1
    T = min(SEQ_LEN, 64 if FAST_MODE else SEQ_LEN)
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    y = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=2**12)
    last = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss = model.forward_loss(x, y, linear_chunk_size=1024, checkpoint_chunks=False)
        if not torch.isfinite(loss):
            raise RuntimeError('%s smoke: loss no finita' % arch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last = float(loss.detach())
    if device.type == 'cuda':
        torch.cuda.synchronize()
        vram = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        vram = 0.0
    del model, opt, scaler, x, y
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return last, vram

print('Humo ...')
for arch in ARCHS:
    t1 = time.time()
    loss, vram = smoke_arch(arch, steps=2 if not FAST_MODE else 1)
    print('  %-22s  loss=%.4f  vram=%.2f GiB  (%.1fs)' % (arch, loss, vram, time.time() - t1))
print('Humo OK.')
"""))

cells.append(md("""## 7. Entrenamiento DDP — los 4 modelos, uno detrás de otro

Cada `torchrun` usa **las dos T4**, el mismo `.ids`, el mismo batch global, LR, warmup y época.

Entre modelos se libera VRAM. Si un job ya dejó `metrics.json` y `RESUME=True`, se reanuda (útil si Kaggle corta la sesión).
"""))

cells.append(code(r"""def run_train(arch, max_steps=0):
    out = os.path.join(SAVE_ROOT, arch)
    os.makedirs(out, exist_ok=True)
    metrics_path = os.path.join(out, 'metrics.json')
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--standalone',
        '--nproc_per_node', str(NPROC),
        worker_path,
        '--arch', arch,
        '--train', train_ids_path,
        '--valid', valid_ids_path,
        '--output', out,
        '--seq-len', str(SEQ_LEN),
        '--vocab-size', str(VOCAB_SIZE),
        '--batch-size', str(LOCAL_BATCH),
        '--eval-batch-size', str(EVAL_BATCH),
        '--epochs', str(EPOCHS),
        '--lr', str(LR),
        '--warmup-steps', str(WARMUP_STEPS),
        '--weight-decay', str(WEIGHT_DECAY),
        '--grad-clip', str(GRAD_CLIP),
        '--log-every', str(LOG_EVERY),
        '--eval-every', str(EVAL_EVERY),
        '--eval-batches', str(EVAL_BATCHES),
        '--linear-chunk-size', str(LINEAR_CHUNK),
        '--max-train-tokens', str(len(train_mm)),
        '--max-valid-tokens', str(len(valid_mm)),
        '--seed', str(SEED),
        '--compile-mode', COMPILE_MODE,
        '--workers', '2' if DEVICE == 'cuda' and not FAST_MODE else '0',
    ]
    if max_steps:
        cmd += ['--max-steps', str(max_steps)]
    if not COMPILE:
        cmd.append('--no-compile')
    if RESUME:
        cmd.append('--resume')
    print('\n>>>', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORK_DIR, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print('FALLO %s  rc=%s  (%.1f min)' % (arch, proc.returncode, elapsed / 60.0))
        return None
    if not os.path.isfile(metrics_path):
        print('FALLO %s: no hay metrics.json' % arch)
        return None
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    metrics['wall_seconds'] = elapsed
    print('OK %s  val=%.4f  ppl=%.2f  %.0f tok/s  %.1f min' % (
        arch,
        metrics.get('best_val_loss') or float('nan'),
        metrics.get('best_val_ppl') or float('nan'),
        metrics.get('tokens_per_sec_steady') or 0.0,
        elapsed / 60.0,
    ))
    return metrics


all_metrics = {}
train_started = time.time()
MAX_STEPS = EST_STEPS if FAST_MODE else 0
for arch in ARCHS:
    print('\n' + '=' * 72)
    print('ENTRENANDO', arch, cards[arch]['title'])
    print('=' * 72)
    try:
        met = run_train(arch, max_steps=MAX_STEPS)
        all_metrics[arch] = met
    except Exception:
        traceback.print_exc()
        all_metrics[arch] = None
    gc.collect()
    if NGPU:
        torch.cuda.empty_cache()

print('\nEntrenamiento total: %.1f min' % ((time.time() - train_started) / 60.0))
with open(os.path.join(SAVE_ROOT, 'all_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, indent=2)
"""))

cells.append(md("""## 8. Cargar checkpoints y helpers de inferencia

A partir de aquí todo corre en **1 GPU** (mediciones comparables, sin DDP).
"""))

cells.append(code(r"""BENCH_DEVICE = torch.device('cuda:0' if NGPU else 'cpu')

def load_trained(arch):
    model = tcd.build_raw_model(arch, VOCAB_SIZE, SEQ_LEN)
    out = Path(SAVE_ROOT) / arch
    weights = out / 'best_model.pt'
    if not weights.is_file():
        weights = out / 'model.pt'
    if not weights.is_file():
        raise FileNotFoundError('No hay pesos para %s en %s' % (arch, out))
    sd = torch.load(str(weights), map_location='cpu')
    model.load_state_dict(sd)
    model.to(BENCH_DEVICE)
    model.eval()
    return model

loaded = {}
for arch in ARCHS:
    if not all_metrics.get(arch):
        print('skip load', arch, '(sin metricas)')
        continue
    loaded[arch] = load_trained(arch)
    print('cargado', arch, 'params', loaded[arch].num_parameters())
"""))

cells.append(md("""## 9. Tokens/s, VRAM vs contexto, orden de complejidad

Se mide el **forward paralelo** (prefill / paso de entrenamiento a batch 1) a longitudes 64, 128, 256, 512 (y 768 si cabe).

Ajuste \(\log t = a + b \log N\):
- \(b \approx 1\) → **O(N)** (ENGRAMA)
- \(b \approx 2\) → **O(N²)** (atención)
- \(b \approx 0\) → **O(1)**

También se mide **decode incremental** (ENGRAMA `step_forward` con caché jerárquica vs Transformer re-prefill) para ver si el coste por token nuevo es constante.
"""))

cells.append(code(r"""LENGTHS = [64, 128, 256, 512] if not FAST_MODE else [32, 64, 128]


def _sync():
    if BENCH_DEVICE.type == 'cuda':
        torch.cuda.synchronize(BENCH_DEVICE)


@torch.no_grad()
def time_forward(model, seq_len, warmup=3, runs=8):
    x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=BENCH_DEVICE)
    amp = BENCH_DEVICE.type == 'cuda'
    for _ in range(warmup):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            _ = model(x)
    _sync()
    if BENCH_DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(BENCH_DEVICE)
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            _ = model(x)
    _sync()
    dt = (time.perf_counter() - t0) / runs
    peak = (torch.cuda.max_memory_allocated(BENCH_DEVICE) / 2**30) if BENCH_DEVICE.type == 'cuda' else 0.0
    return dt, peak


@torch.no_grad()
def time_engrama_step(model, seq_len, warmup=1):
    # Coste medio de un token incremental con cache jerarquica.
    x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=BENCH_DEVICE)
    cache = model.get_cache(N_max=seq_len, mode='hierarchical')
    for t in range(seq_len):
        model.step_forward(x[:, t:t+1], cache, timestamp=t)
    # extra token
    tok = x[:, -1:]
    for _ in range(warmup):
        c2 = model.get_cache(N_max=seq_len + 4, mode='hierarchical')
        for t in range(seq_len):
            model.step_forward(x[:, t:t+1], c2, timestamp=t)
    _sync()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        model.step_forward(tok, cache, timestamp=seq_len)
    _sync()
    return (time.perf_counter() - t0) / reps


@torch.no_grad()
def time_transformer_reprefill(model, seq_len, warmup=2, runs=6):
    # Decode ingenuo: re-forward de toda la secuencia (O(N^2) por token).
    x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=BENCH_DEVICE)
    amp = BENCH_DEVICE.type == 'cuda'
    for _ in range(warmup):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            _ = model(x)
    _sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            _ = model(x)
    _sync()
    return (time.perf_counter() - t0) / runs


def fit_loglog(ns, ys):
    ns = np.asarray(ns, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(ys) & (ys > 0) & (ns > 0)
    if mask.sum() < 2:
        return float('nan'), 'n/a'
    b, a = np.polyfit(np.log(ns[mask]), np.log(ys[mask]), 1)
    if b < 0.4:
        label = 'O(1) ~ plano'
    elif b < 1.4:
        label = 'O(N)'
    elif b < 1.8:
        label = 'O(N log N) / entre N y N²'
    else:
        label = 'O(N²)'
    return float(b), label


scale_report = {}
for arch, model in loaded.items():
    print('\n--- escala', arch, '---')
    times, mems = [], []
    for n in LENGTHS:
        try:
            dt, peak = time_forward(model, n)
        except Exception as exc:
            print('  N=%d FALLO %s' % (n, type(exc).__name__))
            dt, peak = float('nan'), float('nan')
        times.append(dt)
        mems.append(peak)
        print('  N=%4d  forward=%.4fs  (%.0f tok/s)  peak=%.2f GiB' % (
            n, dt if dt == dt else -1, (n / dt) if dt == dt and dt > 0 else 0, peak if peak == peak else -1))
    b_t, lab_t = fit_loglog(LENGTHS, times)
    b_m, lab_m = fit_loglog(LENGTHS, mems)
    step_times = []
    if arch.startswith('engrama'):
        for n in LENGTHS:
            try:
                step_times.append(time_engrama_step(model, n))
            except Exception as exc:
                print('  step N=%d FALLO %s' % (n, type(exc).__name__))
                step_times.append(float('nan'))
        b_s, lab_s = fit_loglog(LENGTHS, step_times)
        print('  step_forward por token: pendiente=%.2f → %s' % (b_s, lab_s))
        print('  tiempos step ms:', ['%.2f' % (1e3 * t) if t == t else 'nan' for t in step_times])
    else:
        b_s, lab_s = fit_loglog(LENGTHS, times)
        print('  re-prefill por token ≈ forward completo: pendiente=%.2f → %s' % (b_t, lab_t))
        step_times = times
    scale_report[arch] = {
        'lengths': LENGTHS,
        'forward_sec': times,
        'peak_gb': mems,
        'forward_slope': b_t,
        'forward_order': lab_t,
        'memory_slope': b_m,
        'memory_order': lab_m,
        'decode_sec': step_times,
        'decode_slope': b_s,
        'decode_order': lab_s,
    }
    print('  forward pendiente=%.2f → %s | memoria pendiente=%.2f → %s' % (b_t, lab_t, b_m, lab_m))

with open(os.path.join(SAVE_ROOT, 'scale_report.json'), 'w', encoding='utf-8') as f:
    json.dump(scale_report, f, indent=2)
"""))

cells.append(md("""## 10. Recuperación clave-valor a largo alcance (zero-shot)

Protocolo tipo MQAR sobre el vocabulario GPT-2:

- 4 pares (clave, valor) aleatorios **por muestra** (no se pueden memorizar globalmente)
- cuerpo de relleno + consultas a distancias ~32, 80, 128, 184 (secuencia 192)
- se puntúa **opción múltiple entre los 16 valores posibles** (azar = 6.25 %) y exact-match sobre 50k

Es una prueba **arquitectónica** (no se fine-tunéa): mide si la señal del par sobrevive en el contexto. V4 debería ganar a las ablaciones; el Transformer atiende \(O(N^2)\) y es una cota alta de recuperación a 192 tokens.
"""))

cells.append(code(r"""KV_SEQ = 192
N_KEYS = 4
KEY_LO, KEY_HI = 1000, 1020
VAL_LO, VAL_HI = 2000, 2015   # 16 valores
FILL_LO, FILL_HI = 200, 250
HEADER_SLOTS = [0, 2, 4, 6]
QUERY_POSITIONS = [32, 80, 128, 184]
N_VALUES = VAL_HI - VAL_LO + 1
CHANCE = 1.0 / N_VALUES


def make_kv_sample(rng):
    keys = rng.sample(range(KEY_LO, KEY_HI + 1), N_KEYS)
    values = [rng.randint(VAL_LO, VAL_HI) for _ in range(N_KEYS)]
    value_of = dict(zip(keys, values))
    seq = [EOS_ID] + [FILL_LO] * (KV_SEQ - 1)
    used = {0}
    for slot, (k, v) in zip(HEADER_SLOTS, zip(keys, values)):
        seq[1 + slot] = k
        seq[1 + slot + 1] = v
        used.add(1 + slot); used.add(1 + slot + 1)
    query_order = keys[:]
    rng.shuffle(query_order)
    answers = []
    for pos, key in zip(QUERY_POSITIONS, query_order):
        seq[pos] = key
        seq[pos + 1] = value_of[key]
        used.add(pos); used.add(pos + 1)
        answers.append((pos, value_of[key]))  # predecir seq[pos+1] desde logits[pos]
    body = [rng.randint(FILL_LO, FILL_HI) for _ in range(8)]
    for i in range(KV_SEQ):
        if i not in used:
            seq[i] = body[i % 8]
    return seq, answers


@torch.no_grad()
def eval_kv(model, n_samples=256 if not FAST_MODE else 32, seed=999):
    rng = random.Random(seed)
    model.eval()
    amp = BENCH_DEVICE.type == 'cuda'
    mc_correct = [0] * N_KEYS
    exact_correct = [0] * N_KEYS
    total = 0
    remaining = n_samples
    value_ids = torch.arange(VAL_LO, VAL_HI + 1, device=BENCH_DEVICE)
    while remaining > 0:
        b = min(16, remaining)
        remaining -= b
        seqs, answers = [], []
        for _ in range(b):
            s, a = make_kv_sample(rng)
            seqs.append(s)
            answers.append(a)
        x = torch.tensor(seqs, dtype=torch.long, device=BENCH_DEVICE)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = model(x[:, :-1])  # (B, T-1, V)
        logits = logits.float()
        for row, ans in enumerate(answers):
            for qi, (pos, value) in enumerate(ans):
                # logits[:, pos] predice el token en pos+1
                row_logits = logits[row, pos]
                exact_correct[qi] += int(int(row_logits.argmax().item()) == value)
                mc_scores = row_logits[value_ids]
                pred = int(value_ids[mc_scores.argmax()].item())
                mc_correct[qi] += int(pred == value)
                total += 1
    out = {
        'overall_mc': sum(mc_correct) / max(1, total),
        'overall_exact': sum(exact_correct) / max(1, n_samples * N_KEYS),
        'chance_mc': CHANCE,
        'n_samples': n_samples,
    }
    for qi in range(N_KEYS):
        dist = QUERY_POSITIONS[qi] - (max(HEADER_SLOTS) + 2)
        out['mc_distance_%d' % dist] = mc_correct[qi] / max(1, n_samples)
        out['exact_distance_%d' % dist] = exact_correct[qi] / max(1, n_samples)
    return out


kv_report = {}
for arch, model in loaded.items():
    print('KV', arch, '...')
    kv_report[arch] = eval_kv(model)
    r = kv_report[arch]
    print('  MC overall=%.1f%% (azar %.1f%%)  exact=%.1f%%' % (
        100 * r['overall_mc'], 100 * r['chance_mc'], 100 * r['overall_exact']))
    for k, v in r.items():
        if k.startswith('mc_distance_'):
            print('   ', k, '= %.1f%%' % (100 * v))

with open(os.path.join(SAVE_ROOT, 'kv_report.json'), 'w', encoding='utf-8') as f:
    json.dump(kv_report, f, indent=2)
"""))

cells.append(md("## 11. Muestras de generación (cualitativo)"))

cells.append(code(r"""PROMPTS = [
    'Once upon a time',
    'One day, a little girl named Anna',
    'Tom found a big red ball',
]


@torch.no_grad()
def generate_ids(model, prompt_ids, max_new=64, temperature=0.8, top_k=40):
    device = BENCH_DEVICE
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    amp = device.type == 'cuda'
    kind = 'engrama' if hasattr(model, 'step_forward') else 'transformer'
    if kind == 'engrama':
        ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature,
                             top_k=top_k, use_cache=True, eos_token_id=EOS_ID)
        return ids
    for _ in range(max_new):
        inp = x[:, -SEQ_LEN:]
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = model(inp)[:, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax(-1).item())
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[-1], torch.full_like(logits, -float('inf')), logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == EOS_ID:
            break
    return x[0].tolist()


samples = {}
for arch, model in loaded.items():
    samples[arch] = {}
    print('\n====', arch, '====')
    for p in PROMPTS:
        ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        try:
            out_ids = generate_ids(model, ids, max_new=48 if FAST_MODE else 80)
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
        except Exception as exc:
            text = '[generacion fallida: %s]' % type(exc).__name__
        samples[arch][p] = text
        print('>', p)
        print(text[:500].replace('\n', ' '), '\n')

with open(os.path.join(SAVE_ROOT, 'samples.json'), 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)
"""))

cells.append(md("## 12. Resumen completo para comparación"))

cells.append(code(r"""def fmt_num(x, nd=4):
    if x is None:
        return '—'
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return '—'
    except Exception:
        return str(x)
    if isinstance(x, float):
        return ('%%.%df' % nd) % x
    return str(x)


def fmt_pct(x):
    if x is None:
        return '—'
    return '%.1f%%' % (100.0 * x)


rows = []
for arch in ARCHS:
    m = all_metrics.get(arch) or {}
    sc = scale_report.get(arch) or {}
    kv = kv_report.get(arch) or {}
    card = (m.get('card') if m else None) or cards.get(arch) or {}
    rows.append({
        'arch': arch,
        'titulo': card.get('title', arch),
        'params_M': (card.get('parameters') or 0) / 1e6,
        'gating': card.get('gating_mode', '—'),
        'trace_tap': card.get('trace_tap', '—'),
        'posicional': card.get('offset_mode') or card.get('positional') or '—',
        'atencion': 'no' if arch.startswith('engrama') else 'sí (SDPA causal)',
        'train_loss': m.get('final_train_loss'),
        'val_loss': m.get('best_val_loss'),
        'val_ppl': m.get('best_val_ppl'),
        'tok_s': m.get('tokens_per_sec_steady'),
        'sec_paso': m.get('sec_per_step'),
        'minutos': m.get('minutes'),
        'vram_train_GiB': m.get('peak_train_vram_gb'),
        'tokens_vistos': m.get('tokens_seen'),
        'skip_nan': m.get('skipped_nonfinite'),
        'forward_orden': sc.get('forward_order'),
        'forward_pendiente': sc.get('forward_slope'),
        'mem_orden': sc.get('memory_order'),
        'mem_pendiente': sc.get('memory_slope'),
        'decode_orden': sc.get('decode_order'),
        'decode_pendiente': sc.get('decode_slope'),
        'kv_mc': kv.get('overall_mc'),
        'kv_exact': kv.get('overall_exact'),
        'kv_d32': kv.get('mc_distance_24') or kv.get('mc_distance_32'),
        'kv_far': None,
    })
    # last distance key
    if kv:
        dist_keys = sorted([k for k in kv if k.startswith('mc_distance_')])
        if dist_keys:
            rows[-1]['kv_d32'] = kv[dist_keys[0]]
            rows[-1]['kv_far'] = kv[dist_keys[-1]]

print('=' * 88)
print('RESUMEN COMPARATIVO  |  TinyStories 100M tok  |  GPT-2 vocab  |  seq 512  |  2xT4 DDP')
print('=' * 88)
hdr = ('%-22s %7s %10s %8s %10s %10s %8s %7s %11s %11s' % (
    'modelo', 'params', 'val_loss', 'ppl', 'tok/s', 's/paso', 'min', 'skip',
    'fwd orden', 'KV-MC'))
print(hdr)
print('-' * 88)
for r in rows:
    print('%-22s %6.2fM %10s %8s %10s %10s %8s %7s %11s %11s' % (
        r['arch'], r['params_M'],
        fmt_num(r['val_loss']), fmt_num(r['val_ppl'], 2),
        fmt_num(r['tok_s'], 0), fmt_num(r['sec_paso'], 3),
        fmt_num(r['minutos'], 1), fmt_num(r['skip_nan'], 0),
        r['forward_orden'] or '—',
        fmt_pct(r['kv_mc']),
    ))

print('\n--- Detalle arquitectónico ---')
print('%-22s %-16s %-10s %-22s %-20s' % ('modelo', 'gating', 'T0 tap', 'offsets/pos', 'atención'))
for r in rows:
    print('%-22s %-16s %-10s %-22s %-20s' % (
        r['arch'], str(r['gating']), str(r['trace_tap']), str(r['posicional']), r['atencion']))

print('\n--- Complejidad empírica (log-log, batch=1, AMP fp16) ---')
print('%-22s %-28s %-28s %-28s' % ('modelo', 'forward vs N', 'VRAM vs N', 'decode/token vs N'))
for r in rows:
    print('%-22s %-28s %-28s %-28s' % (
        r['arch'],
        '%s (b=%.2f)' % (r['forward_orden'] or '—', r['forward_pendiente'] or float('nan')),
        '%s (b=%.2f)' % (r['mem_orden'] or '—', r['mem_pendiente'] or float('nan')),
        '%s (b=%.2f)' % (r['decode_orden'] or '—', r['decode_pendiente'] or float('nan')),
    ))

print('\n--- VRAM forward (GiB) vs contexto ---')
print('%-22s' % 'modelo', end='')
for n in LENGTHS:
    print(' %8s' % ('N=%d' % n), end='')
print()
for arch in ARCHS:
    sc = scale_report.get(arch) or {}
    print('%-22s' % arch, end='')
    for peak in sc.get('peak_gb') or [None] * len(LENGTHS):
        print(' %8s' % (fmt_num(peak, 2) if peak is not None else '—'), end='')
    print()

print('\n--- Recuperación KV (opción múltiple / 16 valores, azar = %.1f%%) ---' % (100 * CHANCE))
print('%-22s %10s %10s %10s %10s' % ('modelo', 'overall', 'cerca', 'lejos', 'exact 50k'))
for r in rows:
    print('%-22s %10s %10s %10s %10s' % (
        r['arch'], fmt_pct(r['kv_mc']), fmt_pct(r['kv_d32']), fmt_pct(r['kv_far']), fmt_pct(r['kv_exact'])))

print('\n--- Entrenamiento ---')
print('%-22s %12s %12s %12s %10s %10s' % (
    'modelo', 'train_loss', 'val_loss', 'val_ppl', 'tok vistos', 'VRAM train'))
for r in rows:
    print('%-22s %12s %12s %12s %10s %10s' % (
        r['arch'], fmt_num(r['train_loss']), fmt_num(r['val_loss']), fmt_num(r['val_ppl'], 2),
        fmt_num(r['tokens_vistos'], 0), fmt_num(r['vram_train_GiB'], 2) + ' GiB'))

print('\nHiperparámetros compartidos:')
print('  dataset=TinyStories GPT-2  train_tokens=%s  valid_tokens=%s  seq=%d  epochs=%d' % (
    format(len(train_mm), ','), format(len(valid_mm), ','), SEQ_LEN, EPOCHS))
print('  nproc=%d  local_batch=%d  global_batch=%d  lr=%g  warmup=%d  cosine  clip=%s  AMP fp16' % (
    NPROC, LOCAL_BATCH, GLOBAL_BATCH, LR, WARMUP_STEPS, GRAD_CLIP))
print('  AdamW betas=(0.9, 0.95)  fused  CE lineal chunk=%d  compile=%s/%s' % (
    LINEAR_CHUNK, COMPILE, COMPILE_MODE))
print('  anti-NaN: CE fp32, GradScaler init_scale=2**12, GEMM fp16 reduction OFF, skip non-finite')

# Ranking helpers
def _key_loss(r):
    v = r['val_loss']
    return v if isinstance(v, (int, float)) and math.isfinite(v) else 1e9

ranked = sorted([r for r in rows if r['val_loss'] is not None], key=_key_loss)
print('\n--- Lectura rápida ---')
if ranked:
    print('  Mejor val_loss/PPL :', ranked[0]['arch'],
          'loss=%s ppl=%s' % (fmt_num(ranked[0]['val_loss']), fmt_num(ranked[0]['val_ppl'], 2)))
tok_ranked = sorted([r for r in rows if r['tok_s']], key=lambda r: r['tok_s'] or 0, reverse=True)
if tok_ranked:
    print('  Más tok/s train    :', tok_ranked[0]['arch'], fmt_num(tok_ranked[0]['tok_s'], 0))
kv_ranked = sorted([r for r in rows if r['kv_mc'] is not None], key=lambda r: r['kv_mc'] or 0, reverse=True)
if kv_ranked:
    print('  Mejor KV-MC        :', kv_ranked[0]['arch'], fmt_pct(kv_ranked[0]['kv_mc']))
print('  Complejidad ENGRAMA (teoría): forward O(N), decode con caché ~O(1) por token (offsets fijos).')
print('  Complejidad Transformer (teoría): forward O(N²), decode con KV-cache O(N) por token.')
print('=' * 88)

summary = {
    'hyper': {
        'seq_len': SEQ_LEN, 'train_tokens': int(len(train_mm)), 'valid_tokens': int(len(valid_mm)),
        'epochs': EPOCHS, 'nproc': NPROC, 'local_batch': LOCAL_BATCH, 'global_batch': GLOBAL_BATCH,
        'lr': LR, 'warmup': WARMUP_STEPS, 'amp': True, 'compile': COMPILE,
    },
    'rows': rows,
    'metrics': all_metrics,
    'scale': scale_report,
    'kv': kv_report,
    'samples': samples,
}
with open(os.path.join(SAVE_ROOT, 'SUMMARY.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)
print('\nArtefactos en', SAVE_ROOT)
for p in sorted(Path(SAVE_ROOT).rglob('*')):
    if p.is_file() and p.stat().st_size < 50 * 1024 * 1024:
        print(' ', p.relative_to(SAVE_ROOT), '%.1f KB' % (p.stat().st_size / 1024))
"""))

cells.append(md("""## 13. Gráficas (loss, tok/s, VRAM, KV)

Si `matplotlib` no está, la celda 12 ya tiene la comparación completa en texto.
"""))

cells.append(code(r"""try:
    import matplotlib.pyplot as plt
except Exception as exc:
    print('matplotlib no disponible:', exc)
else:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for arch in ARCHS:
        hist = (all_metrics.get(arch) or {}).get('history') or []
        if hist:
            ax.plot([h['step'] for h in hist], [h['loss'] for h in hist], label=arch)
    ax.set_title('Train loss'); ax.set_xlabel('paso'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for arch in ARCHS:
        sc = scale_report.get(arch) or {}
        if sc.get('lengths'):
            ax.plot(sc['lengths'], sc['peak_gb'], marker='o', label=arch)
    ax.set_title('VRAM forward vs contexto'); ax.set_xlabel('N'); ax.set_ylabel('GiB')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for arch in ARCHS:
        sc = scale_report.get(arch) or {}
        if sc.get('lengths'):
            ax.loglog(sc['lengths'], sc['forward_sec'], marker='o', label=arch)
    ax.set_title('Forward time vs N (log-log)'); ax.set_xlabel('N'); ax.set_ylabel('s')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')

    ax = axes[1, 1]
    labels, vals = [], []
    for arch in ARCHS:
        kv = kv_report.get(arch) or {}
        if 'overall_mc' in kv:
            labels.append(arch.replace('engrama_', 'e_'))
            vals.append(100 * kv['overall_mc'])
    if vals:
        ax.bar(labels, vals)
        ax.axhline(100 * CHANCE, color='k', ls='--', label='azar')
        ax.set_ylabel('% MC'); ax.set_title('KV retrieval (MC)'); ax.legend()
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig_path = os.path.join(SAVE_ROOT, 'compare_plots.png')
    fig.savefig(fig_path, dpi=120)
    print('fig ->', fig_path)
    plt.show()
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "cells=", len(cells), "bytes=", OUT.stat().st_size)
