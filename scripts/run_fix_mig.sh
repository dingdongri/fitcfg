#!/usr/bin/env bash
# 3차: 병렬 측정에서 실패/오측정한 설정만 다시 (FSDP+LoRA dtype 오류, TP 1B 임베딩 공유 풀림)
#   사용법: cd /workspace/fitcfg && nohup bash scripts/run_fix_mig.sh > /workspace/run_fix.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
python - <<'PY'
import csv
P = "data/gpu_parallel.csv"
rows = list(csv.DictReader(open(P)))
bad = lambda r: (r["kind"] == "fsdp" and r["method"] == "lora") or (r["kind"] == "tp" and r["model"] == "llama-3.2-1b")
keep = [r for r in rows if not bad(r)]
w = csv.DictWriter(open(P, "w", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
print(f"다시 잴 설정 {len(rows) - len(keep)}개 지움 → 남은 {len(keep)}개")
PY
python scripts/gpu_parallel.py run --kinds fsdp tp
python scripts/gpu_parallel.py report || true
tar czf /workspace/results_mig_v3.tar.gz data/gpu_parallel.csv reports/gpu_parallel_report.md
ls -la /workspace/results_mig_v3.tar.gz
echo "=== 전체 완료 $(date '+%H:%M:%S') → 파일 목록에서 workspace/results_mig_v3.tar.gz 다운로드 ==="
