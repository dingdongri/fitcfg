# 소규모 GPU에서 3D 병렬화는 이득인가 (계산 기반 시뮬레이션)

- 시퀀스 2048, global batch 32, 메모리 여유 0.9, GPU당 기본 사용량 1.0GB
- 'DP·FSDP 최선' = TP·PP 없이 데이터 병렬 계열만 쓴 최선, '전체 최선' = TP·PP 포함 탐색
- 이득 = DP·FSDP 최선 시간 ÷ 전체 최선 시간 (1.00×면 3D 병렬화 불필요)
- 통신-계산 겹침 가정 두 가지: 비관(안 겹침) / 낙관(계산 시간만큼 숨겨짐)

| 하드웨어 | 모델 | 방식 | 전체 최선 (낙관) | DP·FSDP 최선 (낙관) | 이득 (낙관) | 이득 (비관) |
|---|---|---|---|---|---|---|
| 4090 ×4 (PCIe) | Llama-8B | full | 불가 | 불가 | — | — |
| 4090 ×4 (PCIe) | Llama-8B | lora | DP2×PP2[16/16] · mbs1 · ckpt off | FSDP4 · mbs2 · ckpt on | 1.17× | 1.30× |
| 4090 ×4 (PCIe) | Llama-8B | qlora | FSDP4 · mbs1 · ckpt off | FSDP4 · mbs1 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.23× |
| 4090 ×4 (PCIe) | Llama-70B | full | 불가 | 불가 | — | — |
| 4090 ×4 (PCIe) | Llama-70B | lora | 불가 | 불가 | — | — |
| 4090 ×4 (PCIe) | Llama-70B | qlora | FSDP2×PP2[40/40] · mbs1 · ckpt on | 불가 | DP·FSDP 불가 → 3D만 가능 | DP·FSDP 불가 → 3D만 가능 |
| 4090 ×8 (PCIe) | Llama-8B | full | FSDP2×PP4[8/8/9/7] · mbs1 · ckpt on | FSDP8 · mbs1 · ckpt on | 1.03× | 1.80× |
| 4090 ×8 (PCIe) | Llama-8B | lora | DP4×PP2[16/16] · mbs1 · ckpt off | FSDP8 · mbs4 · ckpt on | 1.05× | 1.23× |
| 4090 ×8 (PCIe) | Llama-8B | qlora | FSDP8 · mbs1 · ckpt off | FSDP8 · mbs1 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.16× |
| 4090 ×8 (PCIe) | Llama-70B | full | 불가 | 불가 | — | — |
| 4090 ×8 (PCIe) | Llama-70B | lora | PP8[9/10/10/10/11/11/11/8] · mbs1 · ckpt on | 불가 | DP·FSDP 불가 → 3D만 가능 | DP·FSDP 불가 → 3D만 가능 |
| 4090 ×8 (PCIe) | Llama-70B | qlora | FSDP8 · mbs1 · ckpt on | FSDP8 · mbs1 · ckpt on | 1.00× (DP·FSDP가 최적) | 1.17× |
| A100-40 ×4 (NVLink) | Llama-8B | full | TP2×PP2[17/15] · mbs2 · ckpt on | 불가 | DP·FSDP 불가 → 3D만 가능 | DP·FSDP 불가 → 3D만 가능 |
| A100-40 ×4 (NVLink) | Llama-8B | lora | FSDP4 · mbs1 · ckpt off | FSDP4 · mbs1 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.00× (DP·FSDP가 최적) |
| A100-40 ×4 (NVLink) | Llama-8B | qlora | FSDP4 · mbs2 · ckpt off | FSDP4 · mbs2 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.00× (DP·FSDP가 최적) |
| A100-40 ×4 (NVLink) | Llama-70B | full | 불가 | 불가 | — | — |
| A100-40 ×4 (NVLink) | Llama-70B | lora | 불가 | 불가 | — | — |
| A100-40 ×4 (NVLink) | Llama-70B | qlora | FSDP4 · mbs2 · ckpt on | FSDP4 · mbs2 · ckpt on | 1.00× (DP·FSDP가 최적) | 1.00× (DP·FSDP가 최적) |
| A100-80 ×8 (NVLink) | Llama-8B | full | FSDP8 · mbs2 · ckpt off | FSDP8 · mbs2 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.04× |
| A100-80 ×8 (NVLink) | Llama-8B | lora | FSDP8 · mbs4 · ckpt off | FSDP8 · mbs4 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.00× (DP·FSDP가 최적) |
| A100-80 ×8 (NVLink) | Llama-8B | qlora | FSDP8 · mbs4 · ckpt off | FSDP8 · mbs4 · ckpt off | 1.00× (DP·FSDP가 최적) | 1.00× (DP·FSDP가 최적) |
| A100-80 ×8 (NVLink) | Llama-70B | full | 불가 | 불가 | — | — |
| A100-80 ×8 (NVLink) | Llama-70B | lora | FSDP4×TP2 · mbs1 · ckpt off | FSDP8 · mbs4 · ckpt on | 1.19× | 1.15× |
| A100-80 ×8 (NVLink) | Llama-70B | qlora | FSDP4×PP2[41/39] · mbs1 · ckpt off | FSDP8 · mbs4 · ckpt on | 1.20× | 1.21× |
| 3090 ×2 + 4090 ×2 (PCIe, 혼합) | Llama-8B | full | 불가 | 불가 | — | — |
| 3090 ×2 + 4090 ×2 (PCIe, 혼합) | Llama-8B | lora | PP4[5/5/12/10] · mbs1 · ckpt off | FSDP4 · mbs1 · ckpt off | 1.49× | 2.17× |
| 3090 ×2 + 4090 ×2 (PCIe, 혼합) | Llama-8B | qlora | FSDP2×PP2[10/22] · mbs1 · ckpt off | FSDP4 · mbs1 · ckpt off | 1.55× | 1.97× |
| 3090 ×2 + 4090 ×2 (PCIe, 혼합) | Llama-70B | full | 불가 | 불가 | — | — |
| 3090 ×2 + 4090 ×2 (PCIe, 혼합) | Llama-70B | lora | 불가 | 불가 | — | — |
| 3090 ×2 + 4090 ×2 (PCIe, 혼합) | Llama-70B | qlora | PP4[12/13/30/25] · mbs1 · ckpt on | 불가 | DP·FSDP 불가 → 3D만 가능 | DP·FSDP 불가 → 3D만 가능 |
