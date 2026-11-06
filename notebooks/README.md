# Rencana notebook

Notebook berikut belum dibuat. Kerjakan berurutan:

1. `00_data_audit.ipynb`: skema, kualitas dasar, target, ID/waktu, kamus data, dan manifest split.
2. `01_eda_train.ipynb`: analisis train, grafik, interpretasi, dan rekomendasi fitur.
3. `02_preprocessing.ipynb`: pipeline train/validation/test dan pemeriksaan input model.
4. `03_mlp_experiments.ipynb`: MLP, training backpropagation, konfigurasi eksperimen, checkpoint, dan metrik validation.
5. `04_final_evaluation.ipynb`: evaluasi konfigurasi terkunci, analisis error, dan grafik final.

Setiap notebook harus menjelaskan input, output, dan dependensinya, serta dapat dijalankan dari keadaan bersih. Hindari path absolut milik satu anggota. Jika fungsi yang sama dipakai berulang, pertimbangkan memindahkannya ke modul bersama setelah implementasi tersedia.

Simpan angka hasil ke berkas terstruktur; screenshot bukan satu-satunya catatan eksperimen. Detail ada pada bagian 24 `../CONTEXT.md`.
