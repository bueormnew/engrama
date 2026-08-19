import torch
import random
from typing import Tuple, List

VOCAB_SIZE = 64
SEQ_LEN = 2048

KEY_TOKENS = [10, 11, 12, 13]
VAL_TOKENS = [25, 35, 45, 55]
QUERY_POSITIONS = [512, 1024, 1536, 2044]

def generate_sample(seq_len: int = SEQ_LEN) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    seq = torch.zeros(seq_len, dtype=torch.long)
    for t in range(seq_len):
        seq[t] = (t % 8) + 1
        
    bindings = []
    for idx, (k_tok, v_tok) in enumerate(zip(KEY_TOKENS, VAL_TOKENS)):
        seq[idx * 4] = k_tok
        seq[idx * 4 + 1] = v_tok
        bindings.append((k_tok, v_tok))
        
    for idx, (k_tok, v_tok) in enumerate(bindings):
        pos = QUERY_POSITIONS[idx]
        seq[pos] = k_tok
        seq[pos + 1] = v_tok
        
    return seq, bindings

def generate_dataset(num_samples: int, seq_len: int = SEQ_LEN) -> torch.Tensor:
    samples = []
    for _ in range(num_samples):
        seq, _ = generate_sample(seq_len)
        samples.append(seq)
    return torch.stack(samples, dim=0)

def decode_and_verify(model, seq: torch.Tensor, device: str = "cpu") -> float:
    model.eval()
    with torch.no_grad():
        input_seq = seq[:-1].unsqueeze(0).to(device)
        logits = model(input_seq)
        preds = torch.argmax(logits, dim=-1)[0]
        
        correct = 0
        total = len(QUERY_POSITIONS)
        
        for q_pos in QUERY_POSITIONS:
            pred_tok = preds[q_pos].item()
            actual_tok = seq[q_pos + 1].item()
            if pred_tok == actual_tok:
                correct += 1
                
        return correct / total
