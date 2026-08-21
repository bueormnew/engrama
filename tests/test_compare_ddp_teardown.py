"""Guards against the 2xT4 comparison trainer hanging after the first model.

The Kaggle notebook launches four sequential ``torchrun`` jobs. After the last
in-loop eval the worker used to call ``evaluate()`` (an NCCL all_reduce) only on
rank 0 while rank 1 entered ``destroy_process_group`` — a deadlock that looked
like "el primer modelo terminó y nunca arrancó el segundo".

These checks are source-level so they run on CPU CI without torch.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "kaggle" / "train_compare_ddp.py"
NOTEBOOK = ROOT / "kaggle" / "engrama_v4_vs_ablation_transformer_2xt4.ipynb"
GENERATOR = ROOT / "kaggle" / "_gen_compare_notebook.py"


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            src = ast.get_source_segment(path.read_text(encoding="utf-8"), node)
            if src:
                return src
    raise AssertionError(f"function {name} not found in {path}")


class TestCompareDdpTeardown(unittest.TestCase):
    def test_resume_does_not_extend_the_budget(self):
        ns: dict = {}
        exec(_function_source(WORKER, "planned_total_steps"), ns)
        planned_total_steps = ns["planned_total_steps"]
        self.assertEqual(planned_total_steps(6091, 1, 0), 6091)
        self.assertEqual(planned_total_steps(6091, 1, 20), 20)
        self.assertEqual(planned_total_steps(100, 3, 0), 300)
        planned = planned_total_steps(6091, 1, 0)
        start_step = 6091
        self.assertGreaterEqual(start_step, planned)

    def test_final_evaluate_is_not_rank0_only(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertNotRegex(
            source,
            r"if ctx\.is_main:\s+"
            r"torch\.save\(raw_model\.state_dict\(\), output / \"model\.pt\"\)\s+"
            r"save_payload\(raw_model, output, card\)\s+"
            r"val = evaluate\(",
        )
        self.assertIn("if not math.isfinite(last_val):", source)
        self.assertIn("dist.barrier()", source)
        self.assertIn("shutdown_loader", source)
        self.assertIn("planned_total_steps", source)
        self.assertIn("if start_step >= total_steps:", source)

    def test_shutdown_loader_tolerates_none(self):
        ns: dict = {}
        exec(_function_source(WORKER, "shutdown_loader"), ns)
        ns["shutdown_loader"](None)

    def test_train_loop_does_not_sync_every_step(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertNotIn("torch.isfinite(loss.detach())", source)
        self.assertNotIn("loss.detach().float().item()", source)
        self.assertIn("cudagraph_mark_step_begin", source)
        self.assertIn("reduce-overhead", source)
        losses = (ROOT / "src" / "engrama" / "losses.py").read_text(encoding="utf-8")
        self.assertIn("n_tokens <= chunk_size", losses)

    def test_notebook_skips_finished_metrics_and_rewrites_worker(self):
        gen = GENERATOR.read_text(encoding="utf-8")
        nb = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIn("_metrics_complete", gen)
        self.assertIn("--max_restarts", gen)
        self.assertIn("Worker anti-hang escrito", gen)
        self.assertIn("planned_total_steps", nb)
        self.assertIn("_metrics_complete", nb)


if __name__ == "__main__":
    unittest.main()
