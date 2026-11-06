# EV Battery Health — Proyek Jaringan Saraf Tiruan

Proyek kelompok untuk menganalisis dataset kesehatan baterai kendaraan listrik dan menguji **Multi Layer Perceptron (MLP) yang dilatih dengan backpropagation**.

## Status proyek

Tahap saat ini: perencanaan dan persiapan repositori. EDA, preprocessing, pelatihan model, dan hasil eksperimen **belum dijalankan**. CSV belum diaudit; nama target, fitur, satuan, dan struktur waktunya masih perlu diverifikasi.

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

Rekomendasi masalah awal adalah regresi SOH jika target tersebut tersedia dan definisinya jelas. Ini belum merupakan keputusan final berbasis audit CSV.

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
│   └── processed/      # Hasil preprocessing lokal, diabaikan Git
├── notebooks/
│   └── README.md       # Urutan notebook yang direncanakan
├── models/
│   └── README.md       # Aturan penyimpanan model/pipeline
└── results/
    └── README.md       # Aturan pencatatan hasil dan artefak
```

Folder notebook dan model masih berupa kerangka, bukan implementasi yang sudah tersedia.

## Mulai bekerja

```powershell
git clone https://github.com/HUB37T/EV-Battery-Health.git
cd EV-Battery-Health
```

Untuk anggota yang sudah memiliki folder kerja terhubung ke Git, gunakan folder tersebut; tidak perlu clone ulang.

Sebelum membuat model:

1. Baca panduan dan sepakati pembagian kerja.
2. Unduh CSV dan isi lembar keputusan data pada bagian 17 `CONTEXT.md`.
3. Sepakati target, saat prediksi, split, dan fitur yang boleh digunakan.
4. Tentukan apakah dosen mengharuskan backpropagation dari nol atau memperbolehkan framework.
5. Buat lingkungan Python terpisah jika Python dipilih; sepakati versi Python dan dependensi bersama.

Belum ada file dependensi karena framework dan implementasi belum dipilih. Saat implementasi dimulai, catat paket beserta versi yang benar-benar diuji dan tambahkan instruksi eksekusi. Jangan menganggap repositori ini sudah memiliki program training yang dapat dijalankan.

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
