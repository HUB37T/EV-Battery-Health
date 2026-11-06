# Artefak model lokal

Belum ada model prediksi MLP yang dilatih. Fitted preprocessing bundle tersedia lokal di `preprocessing/<recipe>.joblib` setelah notebook 02/modul preprocessing dijalankan. Isi folder ini diabaikan Git kecuali README ini.

Saat pelatihan tersedia, simpan model bersama:
- Pipeline imputasi, encoding, dan scaling yang cocok.
- Urutan fitur, definisi/satuan target, serta timesteps/horizon bila dipakai.
- Konfigurasi arsitektur, seed, versi library, dan identitas run.
- Aturan checkpoint dan epoch terpilih.

Jangan memasukkan model besar ke riwayat Git biasa. Bila model perlu dibagikan, sepakati media penyimpanan artefak dan dokumentasikan cara mendapatkannya. Jangan memuat berkas serialisasi dari sumber yang tidak dipercaya.

Bundle preprocessing menyertakan preprocessor, target scaler, urutan fitur, resep, versi library, provenance, dan hash state. Gunakan `src.preprocessing.load_prepared_split` untuk memeriksa kecocokannya dengan array. Artefak ini dapat dibuat ulang dari data asli dan konfigurasi yang tersimpan; joblib tidak diunggah ke Git.
