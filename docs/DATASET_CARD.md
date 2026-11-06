# Kartu Dataset dan Keputusan Audit

## Sumber

[EV Battery Health & Degradation Dataset — Kaggle](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset).

Versi yang diunduh pada 6 November 2026 diidentifikasi dengan SHA-256 CSV:
`cacdf6afcf50db86583be3754207a89acbfd60b04dfff8f69b71e29cfd78218d`.

Dokumentasi di dalam arsip menyebut **data sepenuhnya sintetis**, generator berbasis model degradasi, serta lisensi **CC0**. Keterangan “physics-informed” adalah penjelasan pembuat, bukan hasil validasi kelompok terhadap data laboratorium. Repositori belum memiliki kode generator atau mask missing/outlier asli.

Arsip yang diterima berisi CSV utama, README, dan `data_dictionary .csv` (ada spasi sebelum ekstensi). Sample notebook yang disebut README tidak terdapat dalam arsip tersebut.

## Fakta yang diverifikasi

- 50.000 observasi, 35 kolom, 2.600 baterai unik.
- Satu baris mewakili satu pengamatan baterai, dengan observation_id unik.
- 7–37 observasi per baterai; angka aktual ini sedikit berbeda dari perkiraan 10–40 dalam README sumber.
- Target utama `state_of_health_pct` tidak kosong dan bersatuan persen.
- Metadata, penggunaan, lingkungan, dan sensor mengandung missing value.
- Data berulang dalam baterai yang sama saling berkaitan; random split per baris tidak dipakai.

Angka audit lengkap dan kamus 35 kolom beserta perannya tersedia pada [laporan audit](../results/audit/AUDIT_REPORT.md) dan [schema_audit.csv](../results/audit/schema_audit.csv).

## Definisi masalah yang digunakan

**Regresi untuk memperkirakan SOH pada waktu observasi, digeneralisasikan ke baterai yang belum ada dalam train.** Ini bukan klaim prediksi sisa umur, kegagalan nyata, atau SOH masa depan.

Asumsi: ringkasan pemakaian/lingkungan dan metadata kandidat fitur telah tersedia pada saat observasi. Sumber tidak merinci interval agregasi fitur; karena itu prediksi sebelum sesi atau forecasting belum didukung.

Split ditetapkan tanpa menggunakan distribusi target: 1.820 baterai train, 390 validation, 390 test. Pemeriksaan seluruh data hanya untuk integritas/kualitas dasar. EDA yang mengarahkan pemilihan fitur menggunakan train.

## Kebijakan fitur awal

Konfigurasi mesin yang menjadi acuan adalah [feature_contract.json](../configs/feature_contract.json). Daftar ini merupakan keputusan awal berbasis definisi sumber, belum hasil seleksi statistik.

- Target: `state_of_health_pct`.
- Kandidat input baseline: 21 kolom, terdiri dari 15 numerik, 5 kategori, dan recall_flag biner.
- ID baterai/observasi: untuk split dan pelacakan, bukan input.
- capacity_fade_pct: dilarang menjadi input karena persis 100 − SOH pada semua nilai terisi.
- remaining_useful_life_cycles, failure_risk_label, degradation_category: target alternatif; dikeluarkan dari input SOH.
- Tanggal mentah: dikeluarkan dari baseline. battery_age_days tetap menjadi kandidat fitur.
- internal_resistance_mohm, voltage_avg_v, current_avg_a, charge_time_avg_min: ditahan sebagai kandidat eksperimen sensor terpisah. README menyebut pengukuran listrik diturunkan dari SOH/kimia dan menyebut resistance secara eksplisit. Mekanisme setiap sensor belum tersedia, sehingga tidak dinyatakan bahwa semuanya pasti mengalami kebocoran dengan cara yang sama.

Pengukuran sensor dapat berguna untuk estimasi saat ini jika tersedia, tetapi hasil dari proxy sintetis tidak boleh disamakan dengan validasi hubungan fisik pada baterai nyata.

## Temuan temporal dan kualitas

- 2.216 observation_date kosong.
- Tidak ada duplikasi pasangan baterai–tanggal ketika tanggal terisi.
- Usia cocok dengan selisih tanggal pada 43.199 baris yang dapat dibandingkan.
- cycle_count dan total_energy_throughput_kwh sering menurun pada tanggal yang lebih baru, meskipun dideskripsikan sebagai kumulatif.
- Missing last_service_date sebagian merupakan ketiadaan struktural karena service_count=0.

Keputusan: baseline tabular; windowing/timesteps ditunda. Data tidak otomatis diperbaiki dengan mengurutkan nilai counter, memaksa monoton, mengisi tanggal masa depan, atau membuang nilai ekstrem.

## Batas kesimpulan

Audit dasar tidak melatih model. Tahap [preprocessing train-only](../results/preprocessing/PREPROCESSING_REPORT.md) kini sudah menerapkan imputasi, encoding, scaling fitur/target, dan dua alternatif resep; study MLP/backpropagation kini tersedia di [laporan eksperimen](../results/experiments/EXPERIMENT_REPORT.md), dengan kandidat yang dipilih dari validation. [Ablation FE](../results/feature_engineering/FEATURE_ENGINEERING_REPORT.md) telah menguji fitur turunan dari input sah; keputusan pra-test mempertahankan kontrol kandidat 03, tanpa fitur domain tambahan. EDA mendalam train sudah tersedia di [laporan EDA](../results/eda/EDA_REPORT.md), mencakup distribusi, hubungan bersyarat, pengamatan berulang, missingness, outlier, dan kualitas temporal. Group split menguji baterai baru pada simulator ini, bukan distribusi dunia nyata atau periode kalender masa depan. [Evaluasi akhir](../results/final_evaluation/FINAL_REPORT.md) kemudian menguji enam checkpoint yang dikunci sebelum test: MAE kandidat 0,95005 ± 0,00619 poin SOH, baseline 1,02684 ± 0,00311 (sample SD tiga seed). Pada SOH <70%, MAE kandidat 3,09049 lebih buruk daripada baseline 2,80057, dengan 88 observasi / 23 baterai. Tidak ada fit/tuning dari test. Hasil ini terbatas pada simulator yang sama; bukan bukti performa dunia nyata, forecasting, atau RUL.

## Reproduksi

Ikuti instalasi di README utama. Dari root proyek:

```powershell
python -m src.data_audit
python scripts/run_notebook.py
```

CSV sumber tidak dimodifikasi. Semua statistik transformasi pada tahap berikutnya harus dipelajari hanya dari train.
