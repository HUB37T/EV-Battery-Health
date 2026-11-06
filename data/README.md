# Data proyek

1. Unduh `ev_battery_health.csv` dari [dataset Kaggle](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset).
2. Simpan file asli di `raw/ev_battery_health.csv`; jangan menimpa dengan hasil pembersihan.
3. Simpan hasil transformasi lokal di `processed/`.

Kedua folder data diabaikan Git, kecuali `.gitkeep`. Jangan mengunggah file kredensial Kaggle.

Saat audit, catat tanggal unduh, versi dataset, lisensi/ketentuan distribusi, jumlah baris/kolom, hash file jika digunakan, definisi target, satuan, ID, dan struktur waktu. Data dan lisensinya belum diverifikasi dalam repositori ini.

Manifest split harus dapat direproduksi melalui seed/aturan split dan identitas data yang sama. Jika manifest berisi data atau identitas yang tidak boleh dibagikan, simpan lokal dan dokumentasikan cara membentuknya kembali.

Panduan audit lengkap tersedia di bagian 17–18 `../CONTEXT.md`.
