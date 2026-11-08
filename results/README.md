# Hasil eksperimen

Audit dasar tersedia di [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md). EDA mendalam tersedia di [eda/EDA_REPORT.md](eda/EDA_REPORT.md), [notebook](../notebooks/01_eda_train.ipynb), dan [HTML baca](eda/EDA_READING.html). Preprocessing tersedia di [preprocessing/PREPROCESSING_REPORT.md](preprocessing/PREPROCESSING_REPORT.md) dan [HTML baca](preprocessing/PREPROCESSING_READING.html). Eksperimen MLP selesai pada validation: [laporan](experiments/EXPERIMENT_REPORT.md), [HTML baca](experiments/EXPERIMENT_READING.html), [rekap run](experiments/run_summary.csv), dan [seed](experiments/seed_summary.csv). [Ablation feature engineering](feature_engineering/FEATURE_ENGINEERING_REPORT.md) membandingkan empat varian dan konfirmasi tiga seed tanpa menimpa studi 03. **Evaluasi test final selesai**: [laporan](final_evaluation/FINAL_REPORT.md), [HTML baca](final_evaluation/FINAL_READING.html), [metrik seluruh seed](final_evaluation/metrics_by_seed.csv), [ringkasan](final_evaluation/metrics_summary.csv), dan [lock sebelum scoring](final_evaluation/evaluation_lock.json). Kandidat MAE 0,95005 ± 0,00619 poin, baseline 1,02684 ± 0,00311; kelemahan kandidat pada SOH <70% tetap dilaporkan. Folder ini menyimpan tabel dan grafik terpilih yang dapat ditelusuri ke data/run nyata.

Untuk setiap run catat:
- ID run dan identitas/versi data serta manifest split.
- Seed, daftar fitur, jumlah input, timesteps/horizon jika relevan.
- Arsitektur, jumlah parameter, aktivasi hidden/output, loss.
- Optimizer, learning rate, batch size, regularisasi.
- Epoch maksimum, aktual, terbaik, serta kebijakan early stopping/checkpoint.
- Metrik train/validation, satuan target, waktu training, dan lokasi artefak.
- Status berhasil/gagal dan alasan jika gagal.

Catat metrik test hanya untuk konfigurasi final yang sudah dikunci. Laporkan semua seed final; jangan memilih seed terbaik dari test. Simpan rerata dan standar deviasi antar-seed dengan arti yang tepat.

Contoh struktur yang dapat dibuat saat ada hasil:
- `experiments.csv`: satu baris per run, bukan angka ilustrasi.
- `figures/`: grafik dengan label/satuan dan ID run.
- `configs/`: konfigurasi yang digunakan.
- `final_metrics.csv`: evaluasi final.

Jangan mengomit dataset, kredensial, atau output mentah berukuran besar di sini. Template tabel laporan ada di bagian 23 `../CONTEXT.md`.

## Hasil final

Folder final_evaluation menyimpan enam gambar, metrik per seed/mean-SD, hasil rentang SOH dan baterai, perbandingan validation–test, serta kontrol konstanta train. Prediksi per observasi disimpan lokal dalam predictions/ dan diabaikan Git. Evaluasi berasal dari enam checkpoint validation yang sudah dikunci; tidak ada model dilatih ulang atau dipilih berdasarkan test.

Untuk regenerasi laporan/gambar dari artefak asli, jalankan `python -m src.final_reporting`. Evaluator memakai cache dengan lock cocok dan menolak penggantian identitas. Reproduksi pada lingkungan baru memakai `python -m src.final_evaluation --reproduce`, dengan keluaran terpisah dalam data/processed; jangan menimpa tabel rilis atau menggunakan test untuk tuning.
