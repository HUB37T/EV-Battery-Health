# Data proyek

1. Unduh `ev_battery_health.csv` dari [dataset Kaggle](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset).
2. Simpan file asli di `raw/ev_battery_health.csv`; jangan menimpa dengan hasil pembersihan.
3. Ekstrak juga `README.md` dan `data_dictionary .csv` ke `raw/`. Nama kamus memiliki spasi sebelum `.csv`.
4. Hasil preprocessing tersedia lokal di `processed/<recipe>/` setelah notebook 02 dijalankan: train/validation/test.npz. Manifest baterai bersama tersedia di `splits/`. Array memuat X, y_scaled/y_original, ID, nama fitur, serta identitas pipeline; jangan diunggah ke Git.

Kedua folder data diabaikan Git, kecuali `.gitkeep`. Jangan mengunggah file kredensial Kaggle.

Saat audit, catat tanggal unduh, versi dataset, lisensi/ketentuan distribusi, jumlah baris/kolom, hash file jika digunakan, definisi target, satuan, ID, dan struktur waktu. README sumber menyebut data sintetis dan lisensi CC0. Audit 6 November 2026 telah memverifikasi skema dan identitas CSV; lihat kartu dataset di `../docs/DATASET_CARD.md`.

Manifest split harus dapat direproduksi melalui seed/aturan split dan identitas data yang sama. Jika manifest berisi data atau identitas yang tidak boleh dibagikan, simpan lokal dan dokumentasikan cara membentuknya kembali.

Panduan audit lengkap tersedia di bagian 17–18 `../CONTEXT.md`.
