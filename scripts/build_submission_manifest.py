# 为可复现交付文件生成 SHA256 清单。

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission" / "MANIFEST.sha256"


# 判断文件是否属于交付清单范围。
def included(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if relative == "submission/MANIFEST.sha256":
        return False
    if "/__pycache__/" in f"/{relative}/" or path.suffix in {".pyc", ".pyo"}:
        return False
    if "/analysis/" in f"/{relative}/":
        return False
    if path.suffix == ".png" and "/fixtures/" in f"/{relative}/":
        return False
    if relative == "submission/innovation/router_incremental.npy":
        return False
    return relative.startswith("submission/") or relative in {
        "README.md",
        "requirements.txt",
        ".gitignore",
    }


# 计算交付文件哈希并写入清单。
def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*") if path.is_file() and included(path)
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
