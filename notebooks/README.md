# Notebook proyek

Notebook 00 dan 01 sudah dibuat dan seluruh selnya berhasil dieksekusi. Notebook 02–04 belum dibuat. Kerjakan berurutan:

1. `00_data_audit.ipynb`: skema, kualitas dasar, target, ID/waktu, kamus data, dan manifest split.
2. `01_eda_train.ipynb`: analisis train, grafik, interpretasi, dan rekomendasi fitur.
3. `02_preprocessing.ipynb`: pipeline train/validation/test dan pemeriksaan input model.
4. `03_mlp_experiments.ipynb`: MLP, training backpropagation, konfigurasi eksperimen, checkpoint, dan metrik validation.
5. `04_final_evaluation.ipynb`: evaluasi konfigurasi terkunci, analisis error, dan grafik final.

Setiap notebook harus menjelaskan input, output, dan dependensinya, serta dapat dijalankan dari keadaan bersih. Hindari path absolut milik satu anggota. Jika fungsi yang sama dipakai berulang, pertimbangkan memindahkannya ke modul bersama setelah implementasi tersedia.

Simpan angka hasil ke berkas terstruktur; screenshot bukan satu-satunya catatan eksperimen. Detail ada pada bagian 24 `../CONTEXT.md`.

Untuk menjalankan ulang audit dari root repositori:

```powershell
python scripts/run_notebook.py
```

Logika bersama berada di `src/data_audit.py`; hasilnya di `results/audit/`. Gunakan lingkungan dengan dependensi `requirements.txt`. Notebook berisi metode, tabel, grafik, interpretasi, serta pemeriksaan integritas; tidak ada model dilatih.

Untuk EDA mendalam:

```powershell
python scripts/run_notebook.py notebooks/01_eda_train.ipynb
python scripts/export_eda.py
```

`01_eda_train.ipynb` berisi 59 sel dengan pertanyaan, metode, tabel, gambar, interpretasi, dan verifikasi. Logika EDA ada di `src/eda.py`, hasil di `results/eda/`. Salinan `EDA_READING.html` dapat dibuka tanpa menjalankan notebook. Tidak ada preprocessing fit atau model pada EDA ini.
