"""In trạng thái tiến độ của khối thực nghiệm chính và khối mở rộng contrastive."""

from __future__ import annotations

import json
from pathlib import Path

CKPT = Path(r"D:\IOT\results\checkpoints")
CKPT_EXT = Path(r"D:\IOT\results\checkpoints_ext")
EXPECTED = 27
EXPECTED_EXT = 24


def _report_dir(path: Path, expected: int, title: str) -> int:
    files = sorted(path.glob("*.json"))
    print(f"\n== {title}: {len(files)}/{expected} lượt ==")
    for f in files:
        j = json.loads(f.read_text(encoding="utf-8"))
        spec = j["spec"]
        print(
            f"  {spec['tag']:<16} val={j['best_val']['macro_f1']:.4f} "
            f"test={j['test']['macro_f1']:.4f} acc={j['test']['accuracy']:.4f} "
            f"epoch={j['best_epoch']:>3} {j['timing']['train_seconds']:6.0f}s"
        )
    missing = expected - len(files)
    print(f"  -> còn thiếu {missing} lượt" if missing else "  -> ĐỦ")
    return len(files)


def main() -> int:
    _report_dir(CKPT, EXPECTED, "Khối chính")
    _report_dir(CKPT_EXT, EXPECTED_EXT, "Khối mở rộng contrastive E3-E6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
