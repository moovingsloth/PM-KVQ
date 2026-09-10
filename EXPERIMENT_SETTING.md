# 실험 설정 — 2026-09-09 day run

| 항목 | 설정 |
|---|---|
| 모델 / GPU | DeepSeek-R1-Distill-Qwen-14B, BF16 / NVIDIA GB10 |
| 비교 | Original → PM-KVQ (`backend=fake`, 양자화 오차 모사) |
| 평가 | AIME 2025 I·II 각각 1–5번, 문제당 4응답: 방법당 40개, 총 80개 |
| 생성 | 최대 32,768토큰, sampling, temperature 0.6, top_p 0.95, seeds 42–45 |
| 보정 | RedPajama 샘플 8개, seq_len 2,048, effective_len 8,192 |
| PM-KVQ | 요청당 KV 예산 1,024 MiB, 할당 후보 4/2bit, 초기 16bit |
| 보호 토큰 | sink 1개 + window 128개, 각각 16bit |
| 환경 | PyTorch 2.9.1+cu130 / Transformers 4.51.3 |

- 실행: `python scripts/smoke_aime2025.py --preset day --wandb`
- 결과·설정 원본: `outputs/day/20260909-171128/` (`metadata.json`)
- 코드: `1a9ca5ecc2f79ed3471e32ddcb3e71ca037f164a`
- [W&B run](https://wandb.ai/dlehddnjs245-kyung-hee-university/pm-kvq/runs/xggofdii)

`fake` backend의 KV 예산은 모사 기준이며, 실제 압축 메모리나 가속 측정값이 아니다.
