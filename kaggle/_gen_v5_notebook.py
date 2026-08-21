#!/usr/bin/env python3
"""Generate kaggle/engrama_v5_long_context_recall.ipynb.

A self-contained notebook that trains ENGRAMA V5 on a synthetic key-value
retrieval task at LONG context (up to 8192 tokens) on a Kaggle GPU and reports
exact-recall accuracy, throughput, and linear-memory generation — the real
8k-token validation that cannot run on the CPU sandbox.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "engrama_v5_long_context_recall.ipynb"


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


CELLS = [
    md(
        "# ENGRAMA V5 — Validación de recuperación en contexto largo (hasta 8192 tokens)\n\n"
        "Este notebook valida en **GPU** lo que el sandbox de CPU no puede: recuperación\n"
        "clave→valor exacta a **contextos enormes**. V5 usa **resonancia sináptica** sobre\n"
        "la traza explícita — **sin atención, sin compresión**.\n\n"
        "Objetivo: **> 85% de recall** a 8000+ tokens.\n"
    ),
    code(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'engrama'], check=False)\n"
        "# Si prefieres el código local del repo:\n"
        "# subprocess.run([sys.executable,'-m','pip','install','-q','-e','/kaggle/input/engrama'], check=False)\n"
        "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    ),
    md("## 1. Definición de la tarea de recuperación (escalable a cualquier longitud)"),
    code(
        "import random, time, math\n"
        "import torch, torch.nn.functional as F\n"
        "\n"
        "VOCAB=64; KEY_LO,KEY_HI=20,35; VAL_LO,VAL_HI=40,55; BOS=0\n"
        "N_KEYS=6                      # 6 pares clave-valor por muestra\n"
        "CHANCE=1.0/(VAL_HI-VAL_LO+1)  # azar\n"
        "\n"
        "def make_sample(rng, SEQ):\n"
        "    header=list(range(1, 1+2*N_KEYS, 2))\n"
        "    q0=max(header)+8\n"
        "    queries=[q0 + i*((SEQ-q0-2)//N_KEYS) for i in range(N_KEYS)]\n"
        "    keys=rng.sample(range(KEY_LO,KEY_HI+1),N_KEYS)\n"
        "    vals=[rng.randint(VAL_LO,VAL_HI) for _ in range(N_KEYS)]\n"
        "    vo=dict(zip(keys,vals))\n"
        "    seq=[BOS]+[rng.randint(4,15) for _ in range(SEQ-1)]\n"
        "    for slot,(k,v) in zip(header,zip(keys,vals)): seq[slot]=k; seq[slot+1]=v\n"
        "    order=keys[:]; rng.shuffle(order); tg=[]\n"
        "    for pos,k in zip(queries,order): seq[pos]=k; seq[pos+1]=vo[k]; tg.append((pos,vo[k]))\n"
        "    return torch.tensor(seq),tg\n"
        "\n"
        "def make_batch(bs,rng,SEQ):\n"
        "    s=[make_sample(rng,SEQ) for _ in range(bs)]\n"
        "    return torch.stack([x for x,_ in s]),[t for _,t in s]\n"
        "print('chance =', f'{CHANCE:.1%}')"
    ),
    md("## 2. Construir el modelo V5"),
    code(
        "from engrama import EngramaV5, EngramaV5Config\n"
        "\n"
        "SEQ = 8192          # <-- contexto largo real\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "cfg = EngramaV5Config(\n"
        "    vocab_size=VOCAB, d_model=256, num_layers=4, num_heads=8,\n"
        "    context_length=SEQ, num_candidates=1, norm_type='layernorm',\n"
        "    chunk_size=512,   # tiling causal: memoria acotada en 8k, idéntico al forward completo\n"
        ")\n"
        "model = EngramaV5(cfg).to(device)\n"
        "print(model.describe())\n"
        "print('params:', f'{model.num_parameters():,}')"
    ),
    md("## 3. Entrenamiento (batch pequeño por la longitud; usa AMP en GPU)"),
    code(
        "BATCH=4; STEPS=4000; ANSWER_WEIGHT=10.0\n"
        "opt=torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01, betas=(0.9,0.95))\n"
        "scaler=torch.amp.GradScaler('cuda', enabled=(device=='cuda'))\n"
        "rng=random.Random(7)\n"
        "\n"
        "def evaluate(SEQ, samples=64):\n"
        "    model.eval(); per=[0]*N_KEYS; ev=random.Random(999)\n"
        "    with torch.no_grad():\n"
        "        left=samples\n"
        "        while left>0:\n"
        "            b=min(BATCH,left); left-=b\n"
        "            seqs,ans=make_batch(b,ev,SEQ)\n"
        "            with torch.amp.autocast('cuda', enabled=(device=='cuda')):\n"
        "                preds=model(seqs[:,:-1].to(device)).argmax(-1)\n"
        "            for r,a in enumerate(ans):\n"
        "                for qi,(pos,v) in enumerate(a): per[qi]+=int(preds[r,pos].item()==v)\n"
        "    model.train(); return sum(per)/(samples*N_KEYS), [p/samples for p in per]\n"
        "\n"
        "model.train(); t0=time.time()\n"
        "for step in range(1,STEPS+1):\n"
        "    seqs,ans=make_batch(BATCH,rng,SEQ)\n"
        "    with torch.amp.autocast('cuda', enabled=(device=='cuda')):\n"
        "        logits=model(seqs[:,:-1].to(device)); tgt=seqs[:,1:].to(device)\n"
        "        raw=F.cross_entropy(logits.reshape(-1,VOCAB),tgt.reshape(-1),reduction='none').view(BATCH,-1)\n"
        "        w=torch.ones_like(raw)\n"
        "        for r,a in enumerate(ans):\n"
        "            for pos,_ in a: w[r,pos]=ANSWER_WEIGHT\n"
        "        loss=(raw*w).mean()\n"
        "    opt.zero_grad(set_to_none=True)\n"
        "    scaler.scale(loss).backward()\n"
        "    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)\n"
        "    scaler.step(opt); scaler.update()\n"
        "    if step % (STEPS//10) == 0:\n"
        "        acc,_=evaluate(SEQ)\n"
        "        print(f'step {step:5d}  loss {loss.item():.3f}  recall {acc:.1%}  ({(time.time()-t0)/step*1000:.0f} ms/step)')"
    ),
    md("## 4. Resultado final por distancia de recuperación"),
    code(
        "acc, per = evaluate(SEQ, samples=128)\n"
        "print(f'RECALL GLOBAL a SEQ={SEQ}: {acc:.1%}  (azar {CHANCE:.1%})')\n"
        "print('por consulta (distancia creciente):', [f'{p:.1%}' for p in per])\n"
        "assert acc > 0.85, 'objetivo >85% no alcanzado — sube STEPS o capacidad'"
    ),
    md("## 5. Garantías: invarianza causal + memoria lineal de generación"),
    code(
        "# Invarianza causal en una secuencia corta\n"
        "model.eval()\n"
        "ids=torch.randint(0,VOCAB,(1,64)).to(device)\n"
        "with torch.no_grad():\n"
        "    par=model.forward(ids)\n"
        "    cache=model.new_cache(); inc=torch.zeros_like(par)\n"
        "    for t in range(64): inc[0,t]=model._step_logits(ids[0,t:t+1],cache)[0]\n"
        "print('invarianza causal |Δ|:', (par-inc).abs().max().item())\n"
        "\n"
        "# Memoria de la caché == O(N) (bytes/token constante)\n"
        "for N in [512,1024,2048,4096]:\n"
        "    c=model.new_cache()\n"
        "    with torch.no_grad():\n"
        "        for t in range(N): model._step_logits(torch.tensor([t%VOCAB]).to(device),c)\n"
        "    print(f'N={N:5d}  bytes/token={c.memory_bytes()//N}')"
    ),
    md(
        "## Conclusión\n\n"
        "Si `RECALL GLOBAL` supera 85% a 8192 tokens, ENGRAMA V5 cumple el requisito de\n"
        "recuperación en contextos enormes **sin atención y sin compresión**, con memoria\n"
        "de generación estrictamente lineal (bytes/token constante) e invarianza causal\n"
        "exacta. Ajusta `STEPS`, `d_model`, `num_layers` si necesitas más margen.\n"
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
