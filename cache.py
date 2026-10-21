"""JSON tuning cache: stable keys, hit/miss tracking, versioning, safe load/save."""
import datetime
import hashlib
import json
import os

CACHE_SCHEMA_VERSION = 1      # bump if the structure of this file's JSON changes


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def build_key_components(spec, n, dtype, env, configs, baseline,
                         algorithm="exhaustive", algo_params=None) -> dict:
    """Everything that could change which config is fastest, or whether a stored
    result can be trusted. If ANY of these differ, the cache must miss."""
    return {
        "kernel": spec.name,
        "kernel_version": spec.version,
        "shape": [int(n)],
        "dtype": str(dtype).replace("torch.", ""),
        "gpu_name": env["gpu_name"],
        "compute_capability": env["compute_capability"],
        "triton_version": env["triton_version"],
        "torch_cuda_build": env["torch_cuda_build"],
        "search_space": sorted([c.block_size, c.num_warps] for c in configs),
        "baseline": [baseline.block_size, baseline.num_warps],
        "algorithm": algorithm,
        "algo_params": algo_params or {},
    }


def make_cache_key(components: dict) -> str:
    """Stable key: readable kernel name + SHA-256 of the canonical JSON of components."""
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{components['kernel']}-{digest}"


def _entry_is_well_formed(entry) -> bool:
    try:
        bc = entry["best_config"]
        return (isinstance(entry["key_components"], dict)
                and isinstance(bc["block_size"], int)
                and isinstance(bc["num_warps"], int)
                and isinstance(entry["measured"], dict)
                and isinstance(entry["search"], dict))
    except (KeyError, TypeError):
        return False


class TuningCache:
    def __init__(self, path: str):
        self.path = path
        self.session_hits = 0
        self.session_misses = 0
        self.data = self._load()

    # ---------------- loading / saving ----------------
    @staticmethod
    def _empty() -> dict:
        return {"schema_version": CACHE_SCHEMA_VERSION, "created_at": now_iso(), "entries": {}}

    def _quarantine(self, reason: str) -> None:
        """Move an unusable cache file aside (never delete user data)."""
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{self.path}.corrupt-{stamp}"
        try:
            os.replace(self.path, backup)
            where = f"Moved the old file to {os.path.basename(backup)}"
        except OSError:
            where = "Could not move the old file"
        print(f"[cache] WARNING: {reason}. {where}; starting with an empty cache.")

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return self._empty()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError) as e:      # JSONDecodeError/UnicodeDecodeError are ValueErrors
            self._quarantine(f"cache file is not valid JSON ({type(e).__name__})")
            return self._empty()
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            self._quarantine("cache file has an unexpected structure")
            return self._empty()
        if data.get("schema_version") != CACHE_SCHEMA_VERSION:
            self._quarantine(f"cache schema version {data.get('schema_version')} "
                             f"!= expected {CACHE_SCHEMA_VERSION}")
            return self._empty()
        return data

    def save(self) -> None:
        """Atomic save: write a temp file, then swap it in."""
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    # ---------------- lookup / insert ----------------
    def lookup(self, components: dict):
        """Return (key, entry_or_None, reason_if_none). Does NOT change counters."""
        key = make_cache_key(components)
        entry = self.data["entries"].get(key)
        if entry is None:
            return key, None, "no entry for this key"
        if not _entry_is_well_formed(entry):
            return key, None, "stored entry is malformed"
        if entry["key_components"] != components:
            return key, None, "stored key components differ (hash collision or edited file)"
        return key, entry, ""

    def record_hit(self, key: str) -> None:
        entry = self.data["entries"][key]
        entry["hits"] = int(entry.get("hits", 0)) + 1
        entry["last_hit_at"] = now_iso()
        self.session_hits += 1
        self.save()

    def record_miss(self) -> None:
        self.session_misses += 1

    def insert(self, key: str, entry: dict) -> None:
        self.data["entries"][key] = entry
        self.save()

    # ---------------- invalidation ----------------
    def invalidate(self, kernel=None, kernel_version=None, gpu_name=None) -> int:
        """Delete entries matching ALL given filters. Returns how many were removed."""
        def matches(entry):
            kc = entry.get("key_components", {}) if isinstance(entry, dict) else {}
            return ((kernel is None or kc.get("kernel") == kernel)
                    and (kernel_version is None or kc.get("kernel_version") == kernel_version)
                    and (gpu_name is None or kc.get("gpu_name") == gpu_name))
        doomed = [k for k, e in self.data["entries"].items() if matches(e)]
        for k in doomed:
            del self.data["entries"][k]
        self.save()
        return len(doomed)

    def clear(self) -> None:
        self.data["entries"] = {}
        self.save()

    # ---------------- reporting ----------------
    def summary_rows(self) -> list:
        rows = []
        for key, e in self.data["entries"].items():
            kc = e.get("key_components", {})
            bc = e.get("best_config", {})
            rows.append({
                "kernel": kc.get("kernel"), "version": kc.get("kernel_version"),
                "n": (kc.get("shape") or [None])[0], "gpu": kc.get("gpu_name"),
                "algorithm": kc.get("algorithm"),
                "best_block": bc.get("block_size"), "best_warps": bc.get("num_warps"),
                "improvement_pct": e.get("measured", {}).get("improvement_pct"),
                "hits": e.get("hits"), "created_at": e.get("created_at"), "key": key,
            })
        return rows
