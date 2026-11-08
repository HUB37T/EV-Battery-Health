# Notebook proyek

Notebook 00–04 termasuk 03b sudah dibuat dan seluruh selnya berhasil dieksekusi. Evaluasi test final selesai dengan enam checkpoint yang dikunci sebelum scoring. Kerjakan berurutan:

1. `00_data_audit.ipynb`: skema, kualitas dasar, target, ID/waktu, kamus data, dan manifest split.
2. `01_eda_train.ipynb`: analisis train, grafik, interpretasi, dan rekomendasi fitur.
3. `02_preprocessing.ipynb`: pipeline train/validation/test dan pemeriksaan input model.
4. `03_mlp_experiments.ipynb`: MLP, training backpropagation, konfigurasi eksperimen, checkpoint, dan metrik validation.
5. `03b_feature_engineering_ablation.ipynb`: ablation fitur domain, kontrol tetap, dan konfirmasi tiga seed pada validation.
6. `04_final_evaluation.ipynb`: evaluasi konfigurasi terkunci, analisis error, dan grafik final.

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

Untuk preprocessing:

```powershell
python scripts/run_notebook.py notebooks/02_preprocessing.ipynb
python scripts/export_eda.py --stage preprocessing
python -m unittest discover -s tests -v
```

Notebook 02 memuat tiga resep, parameter fit train-only, imputasi/encoding/scaling, outlier/log dan feature ablation, kesiapan X/y, uji kasus baru, serta pemakaian bundle. Hasil lokal di data/processed dan models/preprocessing; laporan di results/preprocessing. Belum ada model prediksi yang dilatih atau pemilihan resep menggunakan skor.

Untuk eksperimen MLP:

```powershell
python scripts/run_notebook.py notebooks/03_mlp_experiments.ipynb --timeout 3600
python scripts/export_eda.py --stage experiments
```

Notebook 03 berisi forward/backward, 17 run unik, perbandingan preprocessing/arsitektur/aktivasi/loss/regularisasi, checkpoint best/last, budget epoch 50/100/200, dan konfirmasi tiga seed. Metrik hanya train/validation; test tidak dimuat. Caching memerlukan fingerprint dan checkpoint lokal yang cocok. Kode di src/mlp_experiments.py; laporan/kurva/rekap di results/experiments; checkpoint lokal di models/mlp.

Untuk ablation feature engineering:

```powershell
python scripts/run_notebook.py notebooks/03b_feature_engineering_ablation.ipynb --timeout 3600
python scripts/export_eda.py --stage features
```

Notebook 03b menguji empat varian fitur pada kontrol kandidat 03 yang dibekukan, kemudian mengonfirmasi FE terbaik pada tiga seed. Terdapat 6 run training baru dalam versi final. Hasil/laporan di results/feature_engineering. Tidak mengubah study 03 atau memakai test. Rekomendasi praktis tetap kontrol karena gain agregat FE belum melewati ambang yang ditetapkan.

## Evaluasi final

```powershell
python scripts/run_notebook.py notebooks/04_final_evaluation.ipynb --timeout 3600
python scripts/export_eda.py --stage final
```

Notebook 04 berisi 32 sel (14 sel kode), hasil test tiga seed baseline/kandidat, enam visual, error per rentang SOH dan baterai, verifikasi angka independen, serta suite 32 test. Tidak melatih ulang atau memilih model dari test. MAE kandidat 0,95005 ± 0,00619 poin, baseline 1,02684 ± 0,00311; kandidat lebih buruk pada SOH <70%. Hasil di results/final_evaluation.

Eksekusi memerlukan artefak asli lokal sesuai evaluation_lock.json dan akan menolak hash berbeda. Clone baru dapat langsung membaca laporan/HTML. Jika mereproduksi checkpoint, jalankan evaluator dengan `python -m src.final_evaluation --reproduce` untuk output terpisah, tanpa mengganti lock rilis. Lihat README utama untuk alur lengkap.
