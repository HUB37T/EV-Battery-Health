# Artefak model lokal

Model prediksi MLP sudah dilatih melalui backpropagation dalam study 17 run unik, ditambah 6 run final ablation feature engineering, hanya menggunakan train/validation. Checkpoint best/last serta endpoint epoch tertentu tersedia lokal di `mlp/<run_id>/`; enam best checkpoint validation sudah dipakai untuk evaluasi test final tanpa retraining/fit. Identitas bobot dan preprocessing dicatat sebelum scoring dalam results/final_evaluation/evaluation_lock.json. Fitted preprocessing bundle tersedia lokal di `preprocessing/<recipe>.joblib` setelah notebook 02/modul preprocessing dijalankan. Isi folder ini diabaikan Git kecuali README ini.

Saat pelatihan tersedia, simpan model bersama:
- Pipeline imputasi, encoding, dan scaling yang cocok.
- Urutan fitur, definisi/satuan target, serta timesteps/horizon bila dipakai.
- Konfigurasi arsitektur, seed, versi library, dan identitas run.
- Aturan checkpoint dan epoch terpilih.

Jangan memasukkan model besar ke riwayat Git biasa. Bila model perlu dibagikan, sepakati media penyimpanan artefak dan dokumentasikan cara mendapatkannya. Jangan memuat berkas serialisasi dari sumber yang tidak dipercaya.

Bundle preprocessing menyertakan preprocessor, target scaler, urutan fitur, resep, versi library, provenance, dan hash state. Gunakan `src.preprocessing.load_prepared_split` untuk memeriksa kecocokannya dengan array. Artefak ini dapat dibuat ulang dari data asli dan konfigurasi yang tersimpan; joblib tidak diunggah ke Git.

Checkpoint MLP menyimpan state_dict (bukan seluruh objek model), konfigurasi, nama fitur, epoch, fingerprint input, optimizer, dan RNG. Gunakan `src.mlp_experiments.load_checkpoint` dengan data preprocessing yang cocok. Metadata/hasil run dibagikan di results/experiments, sedangkan bobot tidak dikomit. File checkpoint yang tidak ada pada clone baru dibuat ulang saat study dijalankan. Resume tengah-epoch belum diimplementasikan.

Bundle feature engineering tersimpan lokal di `feature_engineering/<variant>.joblib`. Bundle menggabungkan frozen base preprocessing dan transformasi ekstra; fingerprint parameter tetap sama setelah serialisasi. Array FE hanya train/validation pada tahap 03b. Gunakan `src.feature_engineering.load_engineered_data` untuk memeriksa pasangan pipeline/array sebelum memuat checkpoint MLP yang cocok.

## Model evaluasi final

`configs/final_evaluation.json` menetapkan semua run ID dan seed 42/123/2026. Baseline memakai baseline_standard, 67→[64,32] ReLU→1 linear, MSE; kandidat memakai log_sensor_caps, 67→[32] sigmoid→1 linear, MAE. Kandidat tanpa FE domain ditetapkan sebagai keluarga utama sebelum test. Semua seed dilaporkan, tanpa memilih seed test terbaik atau ensemble.

Bobot tetap lokal dan tidak diubah selama evaluasi. Pipeline cocok diverifikasi dengan fingerprint. Pelatihan ulang pada clone baru dapat menghasilkan hash biner berbeda; gunakan mode `--reproduce` evaluator agar hasil rilis tidak tertimpa. Lock tidak menggantikan kebutuhan menyimpan bobot/pipeline lokal untuk reproduksi bit-identik.
