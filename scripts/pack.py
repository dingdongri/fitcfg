"""Colab/서버로 옮길 zip 만들기: python scripts/pack.py  →  dist/fitcfg.zip
(측정 결과·리포트·캐시는 제외. Colab 노트북의 '업로드' 모드에서 이 파일을 올리면 된다.)"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"data", "reports", "dist", "__pycache__", ".pytest_cache", ".git", "runs", ".ipynb_checkpoints"}


def main():
    out = REPO / "dist" / "fitcfg.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(REPO.rglob("*")):
            rel = p.relative_to(REPO)
            if p.is_dir() or set(rel.parts) & SKIP_DIRS or p.name == ".DS_Store":
                continue
            z.write(p, Path("fitcfg") / rel)
            n += 1
    print(f"{n}개 파일 → {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
