# EV Battery Health — Proyek Jaringan Saraf Tiruan

Proyek kelompok untuk menganalisis dataset kesehatan baterai kendaraan listrik dan menguji **Multi Layer Perceptron (MLP) yang dilatih dengan backpropagation**.

## Status proyek

**Audit dataset dan EDA mendalam train sudah dijalankan.** Data sintetis berisi 50.000 observasi, 35 kolom, dan 2.600 baterai. Target kerja adalah `state_of_health_pct`: estimasi SOH saat observasi pada baterai yang belum ada di train. Preprocessing train-only sudah dijalankan untuk tiga resep kandidat. Resep terbaik belum dipilih; pelatihan dan evaluasi model belum dijalankan.

- [Notebook preprocessing yang sudah dieksekusi](notebooks/02_preprocessing.ipynb).
- [Laporan preprocessing dan hasil pemeriksaan](results/preprocessing/PREPROCESSING_REPORT.md).
- [Salinan baca preprocessing tanpa kode](results/preprocessing/PREPROCESSING_READING.html).
- [Resep preprocessing](configs/preprocessing.json): baseline 67 input; alternatif lengkap 67; compact 63.
- [EDA mendalam: notebook yang sudah dieksekusi](notebooks/01_eda_train.ipynb).
- [Laporan EDA dengan 18 gambar dan interpretasi](results/eda/EDA_REPORT.md).
- [Salinan baca HTML tanpa kode](results/eda/EDA_READING.html) — unduh lalu buka di browser.
- [Notebook audit dasar](notebooks/00_data_audit.ipynb).
- [Ringkasan temuan dan grafik](results/audit/AUDIT_REPORT.md).
- [Kartu dataset dan keputusan metode](docs/DATASET_CARD.md).
- [Kontrak fitur awal](configs/feature_contract.json) dan [manifest split](data/splits/battery_split.csv).

Panduan lengkap, pembagian kerja, teori, dan rancangan eksperimen tersedia di [CONTEXT.md](CONTEXT.md). Mulai dari bagian 17 untuk audit dataset, bagian 21 untuk eksperimen, dan bagian 26 untuk pembagian kerja.

## Ruang lingkup

1. EDA: memahami target, fitur, distribusi, kualitas data, dan potensi kebocoran.
2. Preprocessing: split yang sesuai struktur data, missing value, encoding, normalisasi, seleksi fitur, dan penanganan ketimpangan jika relevan.
3. Eksperimen MLP: variasi input, hidden layer/neuron, aktivasi, loss, dan epoch.
4. Evaluasi: pemilihan konfigurasi menggunakan validation, test final, analisis error, serta pelaporan hasil yang dapat direproduksi.

Model selain MLP tidak termasuk ruang lingkup. Backpropagation menghitung gradien; optimizer memperbarui parameter. Timesteps hanya digunakan jika dataset benar-benar memiliki urutan pengukuran yang sah.

## Dataset

Sumber: [EV Battery Health & Degradation Dataset — Kaggle](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset?select=ev_battery_health.csv).

Unduh `ev_battery_health.csv` dari sumber tersebut dan simpan di:

```text
data/raw/ev_battery_health.csv
```

Data mentah dan data hasil pemrosesan tidak dikomit secara default. Ikuti lisensi dataset dan dokumentasikan tanggal unduh, versi, asal data, serta definisi kolom. Lihat [petunjuk data](data/README.md).

README dalam arsip sumber menyebut data sintetis dan lisensi CC0. Audit membuktikan `capacity_fade_pct = 100 − state_of_health_pct` pada semua nilai terisi; kolom itu dilarang menjadi input. Split menggunakan battery_id: 1.820 baterai train, 390 validation, 390 test. Timesteps ditunda karena tanggal kosong dan counter kumulatif tidak konsisten. Detail ada di kartu dataset.

## Struktur repositori

```text
EV-Battery-Health/
├── CONTEXT.md          # Panduan lengkap kelompok dan rancangan penelitian
├── README.md           # Pengantar dan petunjuk mulai
├── .gitignore          # Data lokal, model, cache, dan kredensial
├── .gitattributes      # Konsistensi akhir baris berkas teks
├── data/
│   ├── README.md       # Cara memperoleh dan mencatat data
│   ├── raw/            # CSV asli lokal, diabaikan Git
│   ├── processed/      # Hasil preprocessing lokal, diabaikan Git
│   └── splits/         # Manifest split baterai bersama
├── configs/            # Kontrak target dan fitur
├── docs/               # Kartu dataset
├── src/                # Logika audit, EDA, dan preprocessing
├── scripts/            # Eksekusi notebook
├── requirements.txt    # Dependensi data/preprocessing yang diuji
├── tests/              # Uji perilaku dan keamanan artefak preprocessing
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_eda_train.ipynb
│   ├── 02_preprocessing.ipynb
│   └── README.md       # Status dan urutan notebook
├── models/
│   └── README.md       # Aturan penyimpanan model/pipeline
└── results/
    ├── audit/          # Audit kualitas dasar dan split
    ├── eda/            # EDA mendalam, PNG/SVG, tabel, HTML
    ├── preprocessing/  # Resep, parameter train, pemeriksaan, HTML
    └── README.md       # Aturan pencatatan hasil dan artefak
```

Notebook audit, EDA, dan preprocessing sudah tersedia dan dieksekusi. Model belum dilatih; folder model masih berupa kerangka.

## Mulai bekerja

```powershell
git clone https://github.com/HUB37T/EV-Battery-Health.git
cd EV-Battery-Health
```

Untuk anggota yang sudah memiliki folder kerja terhubung ke Git, gunakan folder tersebut; tidak perlu clone ulang.

## Menjalankan audit

Lingkungan yang diuji: Python 3.13.1. Dari root proyek:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data_audit
.\.venv\Scripts\python.exe scripts/run_notebook.py
```

CSV, `README.md`, dan `data_dictionary .csv` dari arsip Kaggle harus tersedia di `data/raw/`. Spasi pada nama kamus memang berasal dari arsip. Perintah audit membuat ulang laporan/tabel/grafik; perintah notebook mengeksekusi semua sel dan menyimpan output. CSV asli tidak diubah. Versi CSV yang berbeda dan manifest split yang berubah ditolak. Notebook dapat dibuka di editor notebook dengan lingkungan Python yang sama.

Dependensi ini untuk audit, EDA, dan preprocessing, belum termasuk framework MLP. Matriks versi dan hash sumber ada di `results/audit/provenance.json`.

## Menjalankan EDA mendalam

Dengan lingkungan Python yang sama dan dataset/manifest yang tersedia:

```powershell
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe scripts/run_notebook.py notebooks/01_eda_train.ipynb
.\.venv\Scripts\python.exe scripts/export_eda.py
```

Perintah pertama membuat ulang tabel/grafik/laporan; eksekusi notebook juga membuat ulang hasil tersebut lalu menyimpan output sel. Ekspor HTML memberi salinan baca tanpa kode dengan gambar tertanam. Untuk pekerjaan harian cukup jalankan notebook dan ekspor jika ingin memperbarui salinan HTML.

EDA mencakup semua 35 kolom, 18 gambar, korelasi dengan jumlah pasangan, hubungan kimia–umur berbobot per baterai, bootstrap per baterai, pola bersyarat, missingness, outlier, serta kualitas temporal. EDA hanya menggunakan 34.929 baris / 1.820 baterai train. Data mentah, manifest split, dan kontrak fitur tidak berubah. Detail antarobservasi/per-baterai-bin dihasilkan ulang secara lokal; ringkasan dipublikasikan.

## Menjalankan preprocessing

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py notebooks/02_preprocessing.ipynb
.\.venv\Scripts\python.exe scripts/export_eda.py --stage preprocessing
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Alternatif tanpa notebook: `python -m src.preprocessing`. Semua parameter dipelajari dari train. Array hasil disimpan lokal di `data/processed/<recipe>/` dan fitted bundle di `models/preprocessing/`, diabaikan Git. Laporan/parameter/nama output tersedia di `results/preprocessing/`.

```python
from src.preprocessing import load_prepared_split
train_data, bundle = load_prepared_split('baseline_standard', 'train')
X_train, y_train_scaled = train_data['X'], train_data['y_scaled']
# Setelah prediksi MLP: inverse_transform prediksi memakai bundle['target_scaler'].
```

Pipeline menangani median numerik + indikator missing, kategori missing eksplisit + one-hot, recall biner, standardisasi fitur/target, dan alternatif log/cap/fitur compact. Tidak ada resampling target atau penghapusan SOH rendah. Loader memeriksa kecocokan pipeline/array. Gunakan artefak joblib tepercaya.

Default `baseline_standard` menghasilkan **67 input**; `log_sensor_caps` juga 67; `compact_log_sensor_caps` 63. Ketiganya belum diperingkat berdasarkan skor model. Preprocessing membantu optimasi dan kualitas input, tetapi overfitting/underfitting perlu dibuktikan melalui train–validation curves dan eksperimen MLP. Untuk CV, fit ulang pipeline/target scaler pada setiap fold-train; array precomputed hanya untuk split tetap ini.

## Pekerjaan berikutnya

1. Baca panduan dan sepakati pembagian kerja.
2. Tinjau temuan audit, target, saat prediksi, manifest split, dan kontrak fitur awal bersama kelompok.
3. Tinjau hasil preprocessing dan mulai baseline MLP dengan resep baseline_standard; bandingkan alternatif secara terkontrol menggunakan validation.
4. Tentukan apakah dosen mengharuskan backpropagation dari nol atau memperbolehkan framework.
5. Sesudah pipeline diverifikasi, buat baseline MLP dan eksperimen menggunakan validation. Test tetap untuk evaluasi final.

Belum ada perintah training. Tiga pipeline preprocessing sudah siap; pemilihan resep final, implementasi MLP, dan evaluasi generalisasi masih tahap berikutnya.

## Aturan kerja kelompok

- Gunakan satu definisi target dan manifest split bersama untuk semua eksperimen.
- Fit preprocessing hanya pada train. Gunakan validation untuk memilih konfigurasi dan kunci test sampai evaluasi final.
- Catat konfigurasi, seed, metrik, epoch aktual/terbaik, dan waktu training setiap run.
- Simpan hasil buruk dan run gagal beserta keterangannya.
- Jangan unggah kredensial Kaggle, token, `.env`, lingkungan virtual, atau dataset tanpa memeriksa hak distribusinya.
- Periksa perubahan sebelum commit. Notebook yang akan dibagikan sebaiknya memiliki output ringkas tanpa data sensitif atau dump data besar.
- Jangan mencantumkan skor atau kesimpulan sebelum eksperimennya benar-benar dijalankan.

## Hasil akhir yang dituju

Kamus data dan EDA, pipeline preprocessing, implementasi MLP/backpropagation, tabel eksperimen, evaluasi final, artefak model yang cocok dengan pipeline, laporan, slide, dan instruksi reproduksi. Detail kriteria selesai ada pada bagian 13 dan 30 `CONTEXT.md`.
