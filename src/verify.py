"""Correctness verification. Every outcome (success OR failure) is recorded."""
import time
from dataclasses import dataclass
from typing import Optional

import torch

from config_generator import Config, validate_config


@dataclass
class VerificationResult:
    config: Config
    compiled: bool                      # did the first launch run without error?
    correct: bool                       # did the output match the PyTorch reference?
    max_abs_error: Optional[float]
    first_call_ms: Optional[float]      # wall-clock of 1st call (includes JIT compile)
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict:
        d = self.config.to_dict()
        d.update(compiled=self.compiled, correct=self.correct,
                 max_abs_error=self.max_abs_error, first_call_ms=self.first_call_ms,
                 error_type=self.error_type, error_message=self.error_message)
        return d


def verify_config(spec, inputs, cfg: Config) -> VerificationResult:
    """Launch `spec` once with `cfg`, then compare against spec.reference."""
    ok, reason = validate_config(cfg)
    if not ok:
        return VerificationResult(cfg, False, False, None, None, "InvalidConfig", reason)

    try:
        out = spec.alloc_output(inputs, cfg.block_size)
        out.fill_(float("nan"))         # poison: a kernel that forgets to write will fail
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        spec.launch(inputs, cfg.block_size, cfg.num_warps, out)   # 1st call: compiles
        torch.cuda.synchronize()                                   # wait for the GPU to finish
        first_call_ms = (time.perf_counter() - t0) * 1e3
    except Exception as e:              # compile failure, bad launch config, etc.
        return VerificationResult(cfg, False, False, None, None,
                                  type(e).__name__, str(e)[:300])

    try:
        expected = spec.reference(inputs, cfg.block_size)
        got = out.double()
        exp = expected.double()
        correct = bool(torch.allclose(got, exp, rtol=spec.rtol, atol=spec.atol))
        max_err = float((got - exp).abs().max().item())
        msg = "" if correct else f"output differs from reference (max abs err {max_err:.3e})"
        return VerificationResult(cfg, True, correct, max_err, first_call_ms,
                                  "" if correct else "WrongResult", msg)
    except Exception as e:
        return VerificationResult(cfg, True, False, None, first_call_ms,
                                  type(e).__name__, str(e)[:300])
