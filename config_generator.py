"""Configuration Generator: candidate launch configurations and validation."""
import itertools
from dataclasses import dataclass, asdict

import triton

MAX_TRITON_BLOCK = 1 << 20     # Triton's limit on elements in one tile (tl.arange)
VALID_WARP_COUNTS = {1, 2, 4, 8, 16, 32}


@dataclass(frozen=True, order=True)
class Config:
    """One launch configuration."""
    block_size: int
    num_warps: int

    def label(self) -> str:
        return f"B{self.block_size}_W{self.num_warps}"

    def to_dict(self) -> dict:
        return asdict(self)


def _is_power_of_two(x: int) -> bool:
    return isinstance(x, int) and x > 0 and (x & (x - 1)) == 0


def validate_config(cfg: Config):
    """Return (is_valid, reason). reason is '' when valid."""
    if not _is_power_of_two(cfg.block_size):
        return False, f"block_size={cfg.block_size} is not a positive power of two (required by tl.arange)"
    if cfg.block_size > MAX_TRITON_BLOCK:
        return False, f"block_size={cfg.block_size} exceeds Triton tile limit {MAX_TRITON_BLOCK}"
    if cfg.num_warps not in VALID_WARP_COUNTS:
        return False, f"num_warps={cfg.num_warps} not in {sorted(VALID_WARP_COUNTS)}"
    return True, ""


def generate_configs(block_sizes=(128, 256, 512, 1024), warp_counts=(4, 8)):
    """Cartesian product of the search space.

    Returns (valid_configs, rejected) where `rejected` is a list of
    {"config": Config, "reason": str}. Nothing is dropped silently.
    """
    valid, rejected, seen = [], [], set()
    for b, w in itertools.product(block_sizes, warp_counts):
        cfg = Config(b, w)
        if cfg in seen:
            continue                       # ignore duplicates in user input
        seen.add(cfg)
        ok, reason = validate_config(cfg)
        if ok:
            valid.append(cfg)
        else:
            rejected.append({"config": cfg, "reason": reason})
    return valid, rejected


def grid_size(n_elements: int, cfg: Config) -> int:
    """Number of program instances launched for n_elements."""
    return triton.cdiv(n_elements, cfg.block_size)
