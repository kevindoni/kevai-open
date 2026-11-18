# KevAI-OPEN

Training LoRA otak kecil KevAI 24/7 di GitHub Actions (ubuntu CPU gratis, repo public = menit tanpa batas).

## Rantai 24/7
- Job 6 jam: training + checkpoint tiap 60 step (actions/cache)
- workflow_run: job selesai -> run baru -> RESUME dari checkpoint
- 387/387 tuntas -> merge -> konversi GGUF -> artifact kevai-edge-gguf -> gate DONE

## Privasi
Dataset & kepribadian KevAI PUBLIC di repo ini — keputusan pemilik proyek.
