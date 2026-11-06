# Hasil eksperimen

Audit dasar tersedia di [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md). EDA mendalam tersedia di [eda/EDA_REPORT.md](eda/EDA_REPORT.md), [notebook](../notebooks/01_eda_train.ipynb), dan [HTML baca](eda/EDA_READING.html). Belum ada hasil eksperimen model. Folder ini menyimpan tabel dan grafik terpilih yang dapat ditelusuri ke data/run nyata.

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
