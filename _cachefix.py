"""
Optional: keep every cache / temp file inside this folder (imported before torch).

Off by default. Turn it on by creating an empty file named `.local_cache` next to this
script, or by setting EVIL_QWEN_LOCAL_CACHE=1. Useful when your system drive is full:
ROCm on Windows JIT-compiles GPU kernels and caches them in %LOCALAPPDATA%\\comgr and %TEMP%,
and runs fail with "LLVM ERROR: IO failure on output stream: no space on device" if that drive fills.
"""
import os
from pathlib import Path

R = Path(__file__).resolve().parent

if os.environ.get("EVIL_QWEN_LOCAL_CACHE") == "1" or (R / ".local_cache").exists():
    os.environ.setdefault("HF_HOME", str(R / "hf_cache"))
    os.environ["AMD_COMGR_CACHE"] = "1"
    for k, v in {
        "AMD_COMGR_CACHE_DIR": R / "cache" / "comgr",
        "MIOPEN_USER_DB_PATH": R / "cache" / "miopen",
        "MIOPEN_CUSTOM_CACHE_DIR": R / "cache" / "miopen",
        "PYTORCH_KERNEL_CACHE_PATH": R / "cache" / "torch_kernels",
        "XDG_CACHE_HOME": R / "cache",
        "TMP": R / "tmp", "TEMP": R / "tmp", "TMPDIR": R / "tmp",
    }.items():
        Path(v).mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(v)
