# Laporan Hasil Audit Data dan Exploratory Data Analysis (EDA): Estimasi State of Health (SOH) Baterai Kendaraan Listrik dengan Multilayer Perceptron

---

## BAB I: Pendahuluan dan Konteks Studi Kasus

### 1.1 Latar Belakang dan Rumusan Masalah Studi Kasus

Pengelolaan kesehatan baterai merupakan salah satu aspek krusial dalam operasional kendaraan listrik modern. Seiring bertambahnya usia pakai, frekuensi siklus pengisian daya, dan paparan kondisi lingkungan operasional, kapasitas penyimpanan energi pada paket baterai mengalami penurunan bertahap dibandingkan kapasitas nominalnya saat baru diproduksi. Indikator utama yang merepresentasikan persentase kapasitas aktual baterai terhadap kapasitas nominal awalnya dikenal sebagai *State of Health* (SOH). Dalam studi kasus proyek akhir mata kuliah Jaringan Saraf Tiruan ini, permasalahan diformulasikan sebagai tugas regresi terawasi (*supervised regression*) untuk mengestimasi nilai `state_of_health_pct` (dalam satuan persen) pada saat observasi berlangsung, dengan memanfaatkan ringkasan metrik pemakaian, kondisi lingkungan, dan metadata karakteristik baterai yang tersedia pada waktu pengukuran.

Tujuan generalisasi dari pemodelan ini adalah menghasilkan estimasi SOH yang akurat bagi unit baterai fisik yang belum pernah terlihat sama sekali selama proses pelatihan (*unseen batteries*). Arsitektur model yang digunakan adalah *Multilayer Perceptron* (MLP) dengan algoritma pembelajaran *backpropagation* yang diimplementasikan sepenuhnya dari nol menggunakan pustaka NumPy di dalam modul [model.py](../../src/ev_battery/model.py), mengikuti spesifikasi fungsi `bp_fit` dan `bp_predict` dari modul perkuliahan tanpa bantuan kerangka kerja *deep learning* eksternal. Sebelum merancang skema *preprocessing* dan melatih jaringan saraf tiruan pada [training.py](../../src/ev_battery/training.py), diperlukan tahapan audit data dan *Exploratory Data Analysis* (EDA) yang ketat guna memetakan kualitas data, mengidentifikasi potensi kebocoran informasi (*data leakage*), memahami distribusi serta korelasi antarfitur, dan menentukan kelayakan representasi input bagi jaringan MLP.

### 1.2 Karakteristik Sumber Data dan Verifikasi Provenance

Data yang dianalisis berasal dari repositori publik Kaggle *EV Battery Health & Degradation Dataset* yang dirilis di bawah lisensi CC0 dan disimpan secara lokal pada berkas `data/raw/ev_battery_health.csv` beserta dokumentasi kamus datanya. Berdasarkan dokumentasi pembuat dataset yang dirangkum dalam [DATASET_CARD.md](../DATASET_CARD.md), kumpulan data ini bersifat sepenuhnya sintetis (*fully synthetic*) dan dibangkitkan melalui simulator degradasi baterai yang menginjeksikan nilai hilang (*missing values*) serta pencilan (*outliers*) buatan. Klaim pembuat dataset bahwa sejumlah parameter kelistrikan diturunkan dari nilai SOH dan jenis kimia baterai diperlakukan sebagai hipotesis struktural yang wajib diuji selama tahap audit, mengingat kode program generator maupun *mask* injeksi aslinya tidak disertakan di dalam arsip sumber.

Secara keseluruhan, dataset mentah memuat tepat 50.000 baris observasi dan 35 kolom atribut yang merekam riwayat pembacaan dari 2.600 unit baterai unik (`battery_id`). Untuk menjamin reproduksibilitas ilmiah dan mencegah perubahan data secara tidak sengaja di sepanjang *pipeline*, proses audit pada [00_data_audit.ipynb](../../notebooks/00_data_audit.ipynb) dan [audit.py](../../src/ev_battery/audit.py) memverifikasi sidik jari kriptografis berkas CSV utama menggunakan algoritma SHA-256 sebelum analisis dijalankan. Nilai hash yang terkunci adalah `cacdf6afcf50db86583be3754207a89acbfd60b04dfff8f69b71e29cfd78218d`, dan sistem secara otomatis menghentikan eksekusi apabila ditemukan ketidaksesuaian versi berkas atau perbedaan skema terhadap ke-35 kolom yang terdaftar di kamus data.

### 1.3 Prinsip Metodologi Audit dan Analisis Eksploratif

Pelaksanaan investigasi data dibagi ke dalam dua tahapan berurutan yang terdokumentasi pada [00_data_audit.ipynb](../../notebooks/00_data_audit.ipynb) serta [01_eda.ipynb](../../notebooks/01_eda.ipynb). Tahap pertama, yakni audit data, berfokus pada pemeriksaan integritas skema, keunikan identitas observasi, validitas rentang fisik dasar, pembuktian kebocoran target, uji monotonisitas urutan waktu, dan pembentukan manifest pembagian data (*split manifest*). Pemeriksaan yang menyentuh seluruh 50.000 baris dibatasi secara ketat hanya pada aspek integritas struktural tersebut dan sama sekali tidak digunakan untuk menghitung parameter imputasi, batas pemotongan *outlier*, maupun pemilihan fitur.

Tahap kedua, yakni analisis data eksploratif mendalam pada [01_eda.ipynb](../../notebooks/01_eda.ipynb) dan [eda.py](../../src/ev_battery/eda.py), dijalankan secara eksklusif menggunakan subset pelatihan (*train split*) yang mencakup 34.929 observasi dari 1.820 baterai. Subset *validation* dan *test* diisolasi sepenuhnya agar keputusan perancangan *pipeline* tidak terkontaminasi oleh distribusi data evaluasi. Selain itu, karena satu unit baterai fisik diamati berulang kali dalam rentang 7 hingga 37 kali pembacaan, baris-baris observasi tidak bersifat saling bebas (*non-independent*). Oleh sebab itu, analisis tidak menggunakan nilai *p-value* tingkat baris yang naif, melainkan mengevaluasi pola data melalui dua unit analisis sekaligus: unit observasi (setiap baris memiliki bobot setara) dan unit baterai (setiap `battery_id` memiliki bobot setara).

---

## BAB II: Audit Skema, Integritas Data, dan Kontrak Fitur

### 2.1 Struktur Kamus Data dan Peran Ke-35 Kolom

Pemeriksaan skema pada [00_data_audit.ipynb](../../notebooks/00_data_audit.ipynb) memetakan seluruh 35 kolom di dalam dataset beserta tipe data, satuan pengukuran, jumlah nilai hilang, dan peran fungsionalnya terhadap pemodelan. Secara garis besar, atribut-atribut tersebut terbagi ke dalam kelompok identitas pelacakan, karakteristik statis baterai dan kendaraan, penanggalan peristiwa, akumulasi dan pola pemakaian, kondisi lingkungan operasional, pembacaan sensor kelistrikan, catatan perawatan dan firmware, serta variabel target.

Tabel berikut merangkum struktur ke-35 kolom beserta jumlah nilai hilang pada skala dataset penuh (50.000 baris) dan pada subset *train* (34.929 baris), disertai penetapan peran setiap kolom di dalam kontrak fitur [feature_contract.json](../../configs/feature_contract.json):

| No | Nama Kolom | Tipe Data | Satuan | Missing (Full / %) | Missing (Train) | Unik Non-Null (Train) | Peran dalam Kontrak |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `battery_id` | string | - | 0 (0,000%) | 0 | 1.820 | `excluded` (ID grup split) |
| 2 | `observation_id` | string | - | 0 (0,000%) | 0 | 34.929 | `excluded` (ID pelacakan) |
| 3 | `chemistry` | categorical | - | 3.459 (6,918%) | 2.405 | 4 | `primary_feature` |
| 4 | `manufacturer` | categorical | - | 2.461 (4,922%) | 1.765 | 8 | `primary_feature` |
| 5 | `vehicle_model` | categorical | - | 3.515 (7,030%) | 2.413 | 8 | `primary_feature` |
| 6 | `pack_capacity_kwh` | float | kWh | 2.302 (4,604%) | 1.584 | 550 | `primary_feature` |
| 7 | `manufacture_date` | date | YYYY-MM-DD | 1.910 (3,820%) | 1.356 | 1.225 | `excluded` (tanggal mentah) |
| 8 | `observation_date` | date | YYYY-MM-DD | 2.216 (4,432%) | 1.581 | 2.064 | `excluded` (tanggal mentah) |
| 9 | `battery_age_days` | int | days | 3.002 (6,004%) | 2.115 | 2.064 | `primary_feature` |
| 10 | `cycle_count` | int | cycles | 2.269 (4,538%) | 1.592 | 2.768 | `primary_feature` |
| 11 | `climate_zone` | categorical | - | 2.719 (5,438%) | 1.950 | 5 | `primary_feature` |
| 12 | `avg_daily_distance_km` | float | km | 2.118 (4,236%) | 1.485 | 2.052 | `primary_feature` |
| 13 | `avg_charging_power_kw` | float | kW | 2.140 (4,280%) | 1.513 | 1.475 | `primary_feature` |
| 14 | `fast_charge_ratio` | float | ratio 0-1 | 3.405 (6,810%) | 2.366 | 871 | `primary_feature` |
| 15 | `avg_depth_of_discharge_pct` | float | % | 2.832 (5,664%) | 1.996 | 856 | `primary_feature` |
| 16 | `avg_state_of_charge_pct` | float | % | 3.138 (6,276%) | 2.151 | 607 | `primary_feature` |
| 17 | `charge_cycles_per_month` | float | count/month | 2.054 (4,108%) | 1.426 | 526 | `primary_feature` |
| 18 | `total_energy_throughput_kwh` | float | kWh | 3.370 (6,740%) | 2.362 | 31.799 | `primary_feature` |
| 19 | `avg_ambient_temp_c` | float | derajat C | 3.282 (6,564%) | 2.270 | 1.186 | `primary_feature` |
| 20 | `humidity_pct` | float | % | 2.737 (5,474%) | 1.936 | 941 | `primary_feature` |
| 21 | `altitude_m` | float | meters | 2.322 (4,644%) | 1.632 | 1.587 | `primary_feature` |
| 22 | `internal_resistance_mohm` | float | mOhm | 2.104 (4,208%) | 1.464 | 7.991 | `excluded` (*proxy* sensor) |
| 23 | `voltage_avg_v` | float | V | 2.807 (5,614%) | 1.927 | 702 | `excluded` (*proxy* sensor) |
| 24 | `current_avg_a` | float | A | 2.487 (4,974%) | 1.772 | 1.458 | `excluded` (*proxy* sensor) |
| 25 | `charge_time_avg_min` | float | minutes | 2.091 (4,182%) | 1.413 | 4.659 | `excluded` (*proxy* sensor) |
| 26 | `peak_temp_during_fast_charge_c` | float | derajat C | 2.128 (4,256%) | 1.490 | 1.277 | `primary_feature` |
| 27 | `capacity_fade_pct` | float | % | 1.869 (3,738%) | 1.335 | 3.098 | `excluded` (kebocoran target) |
| 28 | `service_count` | int | count | 1.600 (3,200%) | 1.104 | 15 | `primary_feature` |
| 29 | `last_service_date` | date | YYYY-MM-DD | 18.831 (37,662%) | 13.198 | 2.027 | `excluded` (tanggal mentah) |
| 30 | `bms_firmware_version` | categorical | - | 2.328 (4,656%) | 1.581 | 5 | `primary_feature` |
| 31 | `recall_flag` | binary | 0/1 | 2.646 (5,292%) | 1.845 | 2 | `primary_feature` |
| 32 | `state_of_health_pct` | float | % | 0 (0,000%) | 0 | 3.124 | `target` (target regresi) |
| 33 | `remaining_useful_life_cycles` | int | cycles | 0 (0,000%) | 0 | 4.596 | `excluded` (target alternatif) |
| 34 | `failure_risk_label` | categorical | - | 0 (0,000%) | 0 | 3 | `excluded` (target alternatif) |
| 35 | `degradation_category` | categorical | - | 0 (0,000%) | 0 | 3 | `excluded` (target alternatif) |

Dari pemetaan tersebut, ditetapkan 21 fitur mentah sebagai kandidat input utama (*primary features*), yang terdiri atas 15 fitur numerik, 5 fitur kategorikal, dan 1 fitur biner (`recall_flag`). Tiga belas kolom dikeluarkan dari pemodelan *baseline* dengan alasan metodologis yang spesifik, sedangkan satu kolom ditetapkan sebagai target prediksi utama (`state_of_health_pct`).

### 2.2 Pemeriksaan Integritas Global, Rentang Nilai, dan Konsistensi Atribut Statis

Hasil audit integritas dasar pada keseluruhan dataset membuktikan bahwa struktur tabel bebas dari kerusakan format primer. Tidak ditemukan baris yang terduplikasi secara penuh (`duplicate_rows = 0`), tidak ada duplikasi identitas pembacaan (`duplicate_observation_ids = 0`), tidak ada nilai target yang kosong (`missing_targets = 0`), dan tidak terdapat nilai numerik tak hingga (`infinite_numeric_values = 0`). Pada baris-baris yang memiliki `observation_date` terisi, tidak ditemukan duplikasi pasangan `(battery_id, observation_date)`. Seluruh nilai tanggal yang tidak kosong berhasil diurai (*parsed*) tanpa kesalahan format, dengan rentang waktu `manufacture_date` mulai dari `2020-09-29` hingga `2026-07-29`, `observation_date` dari `2020-10-07` hingga `2026-09-27`, serta `last_service_date` dari `2020-10-11` hingga `2026-09-27`.

Pemeriksaan logika kronologis antartanggal juga menunjukkan hasil yang sepenuhnya konsisten. Tidak ada observasi yang tercatat sebelum tanggal produksi baterai (`observation_before_manufacture = 0`), tidak ada tanggal servis yang melampaui tanggal observasi (`service_after_observation = 0`), dan tidak ada tanggal servis yang mendahului tanggal produksi (`service_before_manufacture = 0`). Lebih lanjut, pada 43.199 baris yang memiliki ketiganya terisi sekaligus (`manufacture_date`, `observation_date`, dan `battery_age_days`), selisih hari antara tanggal observasi dan tanggal produksi cocok secara eksak dengan nilai `battery_age_days` (`age_date_mismatches = 0`). Temuan ini mengonfirmasi bahwa ketiga kolom tanggal mentah tidak perlu dimasukkan ke dalam model karena informasi umur kalender baterai telah terwakili secara utuh oleh `battery_age_days`.

Audit juga menguji batas kewajaran domain pada variabel-variabel berbatas fisik (`state_of_health_pct`, `capacity_fade_pct`, `fast_charge_ratio`, `avg_depth_of_discharge_pct`, `avg_state_of_charge_pct`, `humidity_pct`, dan `recall_flag`), di mana seluruhnya menunjukkan nol pelanggaran rentang (`outside_basic_ranges = 0`). Di samping itu, pengecekan konsistensi internal per baterai membuktikan bahwa keenam atribut statis utama (`chemistry`, `manufacturer`, `vehicle_model`, `pack_capacity_kwh`, `manufacture_date`, dan `climate_zone`) tidak pernah berubah nilai di antara observasi milik baterai yang sama sepanjang nilai tersebut tidak kosong (`within_battery_static_conflicts = 0`).

### 2.3 Identifikasi Kebocoran Target (*Target Leakage*) dan *Proxy* Sensor

Tahapan paling krusial dalam audit fitur adalah mendeteksi kolom yang secara langsung maupun tidak langsung membocorkan nilai target `state_of_health_pct`. Pemeriksaan identitas matematis pada 48.131 baris yang memiliki nilai `capacity_fade_pct` terisi membuktikan bahwa penjumlahan `capacity_fade_pct + state_of_health_pct` selalu menghasilkan angka 100 dengan galat absolut maksimum sebesar `0,0`. Dengan kata lain, `capacity_fade_pct` merupakan komplemen langsung dari SOH (`100 - SOH`). Apabila kolom ini diikutsertakan sebagai fitur input, jaringan MLP hanya akan mempelajari pengurangan linear sederhana dan menghasilkan metrik evaluasi yang tampak sempurna namun gagal total ketika diterapkan pada skenario nyata di mana kapasitas pudar belum diketahui.

Selain `capacity_fade_pct`, tiga kolom target alternatif yakni `remaining_useful_life_cycles`, `failure_risk_label`, dan `degradation_category` wajib dikeluarkan karena ketiganya merupakan label kesehatan turunan yang berada pada posisi hilir yang sama dengan SOH. Keputusan konservatif juga diambil terhadap empat kolom sensor kelistrikan, yaitu `internal_resistance_mohm`, `voltage_avg_v`, `current_avg_a`, dan `charge_time_avg_min`. Dokumentasi sumber menyebutkan bahwa pengukuran listrik di dalam simulator dibangkitkan dari fungsi SOH dan kimia baterai, dengan resistansi internal disebut secara eksplisit. Walaupun dalam sistem manajemen baterai (BMS) nyata sensor tegangan, arus, dan hambatan dalam merupakan input fisik yang sah untuk estimasi SOH, pada dataset sintetis ini keempatnya berisiko bertindak sebagai *proxy* langsung dari formula target generator, sehingga dikeluarkan dari fitur *baseline* dan hanya disimpan untuk pengujian ablasi terpisah. Hubungan awal antara fitur utama serta kandidat *proxy* sensor (`internal_resistance_mohm`) terhadap SOH pada sampel deterministik data *train*, beserta pemeriksaan rentang sebaran sensor menggunakan aturan `1,5 x IQR`, divisualisasikan pada dua grafik hasil audit berikut:

![Hubungan Awal Fitur dan Proxy Sensor terhadap Target SOH pada Data Train](audit_03_relationships_train.png)

![Pemeriksaan Rentang Nilai dan Kandidat Pencilan Sensor pada Data Train](audit_04_sensor_ranges_train.png)

### 2.4 Evaluasi Struktur Longitudinal dan Kelayakan *Timesteps*

Karena setiap baterai memiliki rangkaian observasi berulang, audit menguji apakah dataset ini layak diformat sebagai deret waktu (*timesteps* atau *sliding window*). Hasil penghitungan frekuensi menunjukkan bahwa jumlah observasi per baterai berkisar dari minimum 7 hingga maksimum 37 pembacaan (median 19,000, kuartil bawah 16,000, kuartil atas 22,000, rerata 19,231, dan simpangan baku 4,367), sedikit berbeda dari klaim kasar dokumentasi sumber yang menyebutkan 10 hingga 40 observasi.

Untuk menilai kelayakan penyusunan urutan waktu, dilakukan pengujian monotonisitas pada variabel-variabel kumulatif terhadap pasangan observasi berurutan dari baterai yang sama yang memiliki tanggal valid dan menaik:

| Variabel Kumulatif | Pasangan Observasi Valid (*Eligible Pairs*) | Pasangan dengan Nilai Menurun (*Decreasing Pairs*) | Persentase Penurunan (%) |
|---|---:|---:|---:|
| `cycle_count` | 43.017 | 7.820 | 18,179% |
| `battery_age_days` | 42.320 | 0 | 0,000% |
| `total_energy_throughput_kwh` | 41.948 | 8.923 | 21,272% |

Tabel di atas memperlihatkan anomali serius pada struktur longitudinal simulator. Meskipun `battery_age_days` selalu bertambah seiring kenaikan tanggal observasi (0% penurunan), dua penghitung yang secara fisik bersifat kumulatif justru sering mengalami penurunan nilai pada tanggal yang lebih baru: `cycle_count` turun pada 7.820 dari 43.017 pasangan (18,179%), sedangkan `total_energy_throughput_kwh` turun pada 8.923 dari 41.948 pasangan (21,272%). Ditambah dengan fakta bahwa terdapat 2.216 baris tanpa `observation_date`, pembentukan *timesteps* akan menuntut penghapusan data secara masif atau rekayasa pengurutan paksa yang mengubah struktur asli data. Oleh karena itu, diputuskan bahwa model menggunakan representasi tabular satu baris per observasi tanpa *timesteps*.

### 2.5 Strategi Pembagian Data Berbasis Grup Baterai (*Group Split* 70/15/15)

Mengingat observasi dari baterai yang sama memiliki korelasi internal yang tinggi, pembagian data secara acak per baris (*random row split*) dilarang keras karena akan menempatkan pembacaan dari unit baterai yang sama di dalam himpunan *train* sekaligus *test*. Untuk menguji kemampuan generalisasi jaringan saraf tiruan terhadap baterai baru secara jujur, pembagian dilakukan pada tingkat `battery_id` dengan rasio 70% *train*, 15% *validation*, dan 15% *test*. Algoritma pembagian pada `build_split` mengurutkan seluruh 2.600 `battery_id` secara leksikografis, mengacak urutannya secara deterministik menggunakan generator `numpy.random.default_rng(42)` tanpa melihat nilai target, lalu memetakan setiap grup baterai ke masing-masing subset:

| Subset (*Split*) | Jumlah Baterai Unik | Jumlah Baris Observasi | Persentase Baris (%) |
|---|---:|---:|---:|
| **Train** | 1.820 | 34.929 | 69,858% |
| **Validation** | 390 | 7.499 | 14,998% |
| **Test** | 390 | 7.572 | 15,144% |
| **Total** | **2.600** | **50.000** | **100,000%** |

Verifikasi himpunan membuktikan bahwa irisan `battery_id` antarketiga subset adalah himpunan kosong (tidak ada tumpang tindih baterai) dan manifest pembagian tersebut dikunci ke dalam berkas `data/splits/battery_split.csv`. Setelah pembagian dikunci, profil awal pada himpunan *train* langsung dicatat pada tahap audit untuk memetakan tingkat *missingness* per kolom serta proporsi awal distribusi target SOH dan kimia baterai pada tingkat observasi:

![Profil Awal Persentase Nilai Hilang pada Data Train Hasil Audit](audit_01_missing_train.png)

![Profil Awal Distribusi Target SOH dan Komposisi Kimia Baterai pada Data Train](audit_02_target_chemistry_train.png)

---

## BAB III: Distribusi Target, Struktur Observasi, dan Profil Fitur pada Data *Train*

### 3.1 Distribusi Target *State of Health* (SOH) dan Ketimpangan Rentang

Memasuki tahap *Exploratory Data Analysis* pada [01_eda.ipynb](../../notebooks/01_eda.ipynb), fokus analisis diarahkan sepenuhnya pada 34.929 baris dari 1.820 baterai *train*. Variabel target `state_of_health_pct` memiliki nilai minimum 55,000% dan nilai maksimum 100,000%, dengan nilai rerata (*mean*) sebesar 92,919%, simpangan baku (*standard deviation*) sebesar 7,193%, dan median (persentil ke-50) sebesar 95,230%. Nilai kuartil bawah (persentil ke-25) berada pada 90,080% dan kuartil atas (persentil ke-75) berada pada 98,050%, sedangkan persentil ke-1 dan ke-5 masing-masing tercatat pada 67,256% dan 78,220%. Koefisien kemiringan (*skewness*) target sebesar `-1,808` mengonfirmasi bahwa distribusi SOH sangat miring ke kiri (*left-skewed*), dengan mayoritas observasi terkonsentrasi pada kondisi baterai yang masih sangat sehat dan ekor panjang menjulur ke arah SOH rendah.

![Distribusi Target SOH dan Pembobotan per Baterai](01_target_and_weighting.png)

Terdapat fenomena penumpukan nilai (*pile-up*) tepat pada batas atas dan batas bawah rentang pengamatan, yakni sebanyak 1.603 baris bernilai tepat 100,000% dan 80 baris bernilai tepat 55,000%. Pola ini mengindikasikan adanya pemotongan batas (*clipping*) atau pembulatan di dalam logika simulator pembuat data. Ketimpangan proporsi sampel antarrentang kesehatan baterai terlihat jelas ketika data *train* dikelompokkan ke dalam lima pita SOH (*SOH bands*):

| Pita SOH (`soh_band_pct`) | Jumlah Baris *Train* (`train_rows`) | Persentase Observasi *Train* (`train_pct`) | Konteks Domain |
|---|---:|---:|---|
| `< 70%` | 525 | 1,503% | Degradasi berat (sangat jarang) |
| `70% - < 80%` | 1.697 | 4,858% | Di bawah ambang *end-of-life* (80%) |
| `80% - < 90%` | 6.412 | 18,357% | Degradasi menengah |
| `90% - < 95%` | 8.264 | 23,659% | Degradasi ringan |
| `>= 95%` | 18.031 | 51,622% | Kondisi sehat / mendekati baru |

Lebih dari separuh data pelatihan (51,622%) berada pada rentang SOH `>= 95%`, sementara total observasi yang berada di bawah batas *end-of-life* 80% hanya mencakup 6,361% (gabungan 4,858% pada pita `70% - < 80%` dan 1,503% pada pita `< 70%`). Karena target bersifat kontinu, metode penyeimbangan kelas diskret seperti SMOTE tidak dapat diterapkan. Namun, ketimpangan ini memberikan implikasi penting bahwa metrik MAE agregat berisiko didominasi oleh baterai sehat, sehingga evaluasi akhir wajib melaporkan galat prediksi secara terpisah untuk setiap pita SOH serta menguji skema pembobotan *loss* berbasis frekuensi pita SOH (*sample weighting*).

### 3.2 Evaluasi Bias Jumlah Observasi per Baterai

Mengingat jumlah observasi per baterai bervariasi antara 7 hingga 37 baris, terdapat potensi bias apabila baterai yang memiliki frekuensi pengamatan lebih banyak ternyata cenderung lebih sehat atau lebih terdegradasi. Untuk menguji apakah variasi jumlah baris per baterai mendistorsi profil target, distribusi SOH dihitung ulang dengan memberikan bobot `1 / jumlah_observasi_baterai` pada setiap baris sehingga setiap unit baterai memiliki kontribusi total bobot yang sama (tepat 1,0):

| Metrik Evaluasi | Nilai Terukur |
|---|---:|
| Rerata (*Mean*) Bobot Sama per Observasi | 92,919% |
| Rerata (*Mean*) Bobot Sama per Baterai | 92,913% |
| Median Bobot Sama per Observasi | 95,230% |
| Median Bobot Sama per Baterai | 95,210% |
| Korelasi Spearman (Jumlah Observasi vs Rerata SOH Baterai) | 0,014 |

Perbandingan di atas menunjukkan bahwa selisih antara statistik tingkat observasi dan statistik tingkat baterai sangat kecil (selisih rerata hanya 0,006 poin persentase dan selisih median hanya 0,020 poin persentase). Koefisien korelasi peringkat Spearman antara jumlah observasi per baterai dan rerata SOH masing-masing baterai juga mendekati nol (`0,014`), sebagaimana divisualisasikan pada grafik struktur observasi baterai berikut:

![Struktur Jumlah Observasi per Baterai dan Hubungannya dengan Rerata SOH](16_battery_observation_structure.png)

Dengan demikian, perbedaan jumlah baris per baterai tidak menimbulkan bias sistematis pada ringkasan distribusi target secara global. Walaupun demikian, karena pembacaan di dalam satu baterai tetap saling berkorelasi, pelaporan performa model pada tingkat baterai (rerata galat per `battery_id`) tetap dipertahankan sebagai pelengkap metrik tingkat observasi.

### 3.3 Karakteristik Distribusi Fitur Numerik dan Ketimpangan Skala

Eksplorasi terhadap ke-15 fitur numerik kandidat utama, atribut biner `recall_flag`, serta keempat kolom *proxy* sensor mengungkap keberagaman bentuk distribusi dan rentang skala yang sangat lebar di dalam himpunan *train*. Visualisasi histogram tanpa pemotongan nilai ekstrem disajikan pada dua panel distribusi berikut:

![Distribusi Fitur Numerik pada Data Train Bagian 1](02_numeric_distributions_1.png)

![Distribusi Fitur Numerik pada Data Train Bagian 2](02_numeric_distributions_2.png)

Tabel statistik deskriptif berikut merangkum jumlah data teramati (`count`), rerata, simpangan baku, kuantil, nilai ekstrem, serta tingkat kemiringan (*skewness*) untuk seluruh variabel numerik pada data *train*:

| Nama Fitur | Count | Mean | Std | Min | P1 | P25 | Median (P50) | P75 | P99 | Max | Skewness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pack_capacity_kwh` | 33.345 | 72,631 | 14,400 | 40,100 | 41,700 | 62,500 | 72,500 | 82,500 | 99,400 | 99,900 | -0,098 |
| `battery_age_days` | 32.814 | 555,565 | 476,557 | 1,000 | 6,000 | 167,000 | 418,000 | 835,000 | 1.888,870 | 2.184,000 | 0,992 |
| `cycle_count` | 33.337 | 572,371 | 606,371 | 1,000 | 4,000 | 136,000 | 372,000 | 800,000 | 2.802,640 | 5.073,000 | 1,918 |
| `avg_daily_distance_km` | 33.444 | 50,487 | 39,826 | 3,000 | 4,700 | 22,500 | 39,300 | 66,800 | 192,657 | 400,000 | 1,825 |
| `avg_charging_power_kw` | 33.416 | 31,975 | 17,701 | 1,496 | 4,200 | 19,000 | 28,600 | 41,800 | 85,385 | 147,772 | 0,997 |
| `fast_charge_ratio` | 32.563 | 0,293 | 0,172 | 0,000 | 0,000 | 0,161 | 0,272 | 0,405 | 0,746 | 0,937 | 0,567 |
| `avg_depth_of_discharge_pct` | 32.933 | 64,754 | 15,820 | 10,000 | 26,300 | 54,100 | 65,000 | 75,900 | 100,000 | 100,000 | -0,146 |
| `avg_state_of_charge_pct` | 32.778 | 67,624 | 10,127 | 30,900 | 43,300 | 60,800 | 67,900 | 74,600 | 90,300 | 95,000 | -0,116 |
| `charge_cycles_per_month` | 33.503 | 20,307 | 10,289 | 2,000 | 2,000 | 12,800 | 19,000 | 26,600 | 47,300 | 57,500 | 0,568 |
| `total_energy_throughput_kwh` | 32.567 | 38.685,231 | 42.928,649 | 36,400 | 292,750 | 8.772,200 | 24.155,300 | 53.217,100 | 200.447,024 | 402.419,900 | 2,178 |
| `avg_ambient_temp_c` | 32.659 | 21,135 | 12,174 | -23,400 | -7,900 | 13,000 | 21,500 | 29,600 | 47,600 | 83,715 | -0,123 |
| `humidity_pct` | 32.993 | 54,915 | 17,830 | 5,000 | 13,492 | 42,700 | 54,800 | 67,000 | 96,500 | 100,000 | -0,008 |
| `altitude_m` | 33.297 | 298,664 | 296,691 | 0,000 | 3,000 | 86,000 | 209,000 | 416,000 | 1.357,040 | 3.214,000 | 1,955 |
| `peak_temp_during_fast_charge_c` | 33.439 | 26,705 | 12,771 | -10,904 | -2,600 | 18,100 | 26,900 | 35,550 | 55,186 | 89,820 | -0,041 |
| `service_count` | 33.825 | 1,415 | 1,664 | 0,000 | 0,000 | 0,000 | 1,000 | 2,000 | 7,000 | 15,000 | 1,619 |
| `recall_flag` (biner) | 33.084 | - | - | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 1,000 | 1,000 | 7,795 |
| `internal_resistance_mohm` (*proxy*) | 33.465 | 49,298 | 21,678 | 7,677 | 24,280 | 34,800 | 42,920 | 56,990 | 128,471 | 307,090 | 2,046 |
| `voltage_avg_v` (*proxy*) | 33.002 | 363,925 | 12,257 | 318,500 | 335,200 | 355,200 | 365,600 | 372,975 | 387,700 | 401,700 | -0,354 |
| `current_avg_a` (*proxy*) | 33.157 | 137,643 | 20,016 | 25,347 | 92,700 | 125,000 | 137,100 | 150,100 | 184,988 | 291,816 | 0,213 |
| `charge_time_avg_min` (*proxy*) | 33.516 | 126,249 | 108,855 | 5,471 | 21,200 | 60,500 | 94,400 | 151,700 | 650,170 | 1.190,084 | 2,807 |

Dari tabel di atas, terlihat dua karakteristik fundamental yang menentukan desain *preprocessing*. Pertama, rentang skala antarfitur berbeda hingga beberapa orde besaran: `fast_charge_ratio` bergerak di antara `0,000` hingga `0,937`, sedangkan `total_energy_throughput_kwh` mencapai `402.419,900` kWh dengan simpangan baku `42.928,649` kWh. Memasukkan fitur-fitur mentah ini secara langsung ke dalam MLP akan membuat pembaruan gradien didominasi oleh fitur berskala puluhan ribu dan menyebabkan saturasi dini pada fungsi aktivasi sigmoid maupun tanh. Oleh sebab itu, standardisasi (`StandardScaler`) atau penskalaan min-max (`MinMaxScaler`) wajib diterapkan pada tahap *preprocessing*. Kedua, sejumlah fitur numerik utama memiliki ekor kanan yang sangat panjang (*positive skewness* mendekati atau melebihi 1,8), yaitu `total_energy_throughput_kwh` (2,178), `altitude_m` (1,955), `cycle_count` (1,918), `avg_daily_distance_km` (1,825), dan `service_count` (1,619), yang memotivasi pengujian transformasi logaritmik (`log1p`) pada resep *preprocessing* alternatif.

### 3.4 Profil Fitur Kategorikal dan Atribut Biner

Selain variabel numerik kontinu, kontrak fitur *baseline* mencakup lima atribut kategorikal (`chemistry`, `manufacturer`, `vehicle_model`, `climate_zone`, dan `bms_firmware_version`) serta satu atribut biner (`recall_flag`). Dalam analisis eksploratif ini, baris yang memiliki nilai kosong pada kolom kategorikal dikelompokkan secara eksplisit sebagai kategori `Missing` agar proporsi ketiadaan data terlihat jelas dan tidak disamarkan ke dalam modus kategori terbanyak.

![Frekuensi Observasi dan Dukungan Baterai pada Fitur Kategorikal serta Recall Flag](03_categorical_support.png)

Distribusi frekuensi baris, jumlah baterai unik yang terwakili, persentase baris, serta median SOH dan median umur untuk masing-masing kategori pada data *train* dirangkum dalam tabel berikut:

| Fitur | Kategori | Baris (*Rows*) | Baterai Unik | Persentase Baris (%) | Median SOH (%) | Median Umur (Hari) | Median Siklus |
|---|---|---:|---:|---:|---:|---:|---:|
| `chemistry` | NMC | 10.884 | 613 | 31,160% | 94,760 | 417,0 | 375,0 |
| `chemistry` | LFP | 9.805 | 546 | 28,071% | 97,350 | 418,0 | 365,0 |
| `chemistry` | NCA | 6.985 | 393 | 19,998% | 94,040 | 425,0 | 384,0 |
| `chemistry` | LMO | 4.850 | 268 | 13,885% | 92,090 | 408,0 | 360,5 |
| `chemistry` | Missing | 2.405 | 1.345 | 6,885% | 95,200 | 429,5 | 374,0 |
| `manufacturer` | GigaWatt Labs | 4.670 | 260 | 13,370% | 95,405 | 398,0 | 340,0 |
| `manufacturer` | CellTech | 4.309 | 235 | 12,336% | 94,370 | 479,0 | 456,0 |
| `manufacturer` | PowerGrid Batteries | 4.278 | 236 | 12,248% | 95,215 | 400,0 | 368,0 |
| `manufacturer` | IonFlux | 4.192 | 228 | 12,001% | 94,965 | 423,0 | 372,0 |
| `manufacturer` | NexCell | 4.053 | 226 | 11,604% | 95,240 | 430,0 | 362,0 |
| `manufacturer` | VoltaCore | 3.987 | 219 | 11,415% | 95,620 | 381,0 | 356,0 |
| `manufacturer` | Amperion | 3.980 | 216 | 11,395% | 95,440 | 426,0 | 360,0 |
| `manufacturer` | StormCell | 3.695 | 200 | 10,579% | 95,450 | 422,0 | 364,0 |
| `manufacturer` | Missing | 1.765 | 1.145 | 5,053% | 95,340 | 420,0 | 373,0 |
| `vehicle_model` | Van-E4 | 4.360 | 238 | 12,482% | 95,480 | 415,0 | 364,0 |
| `vehicle_model` | Compact-E1 | 4.171 | 230 | 11,941% | 94,980 | 445,0 | 388,0 |
| `vehicle_model` | SUV-E5 | 4.120 | 234 | 11,795% | 95,230 | 391,0 | 365,5 |
| `vehicle_model` | Hatch-E3 | 4.097 | 231 | 11,730% | 95,910 | 379,0 | 327,0 |
| `vehicle_model` | Crossover-X2 | 3.996 | 225 | 11,440% | 95,070 | 433,0 | 402,5 |
| `vehicle_model` | Pickup-E7 | 3.980 | 223 | 11,395% | 94,955 | 419,0 | 367,0 |
| `vehicle_model` | Sport-E9 | 3.920 | 223 | 11,223% | 94,890 | 429,0 | 396,5 |
| `vehicle_model` | Sedan-E1 | 3.872 | 216 | 11,085% | 95,310 | 442,0 | 366,0 |
| `vehicle_model` | Missing | 2.413 | 1.339 | 6,908% | 95,140 | 421,0 | 367,0 |
| `climate_zone` | hot_arid | 6.967 | 379 | 19,946% | 95,220 | 414,0 | 379,0 |
| `climate_zone` | tropical_humid | 6.807 | 375 | 19,488% | 95,020 | 422,0 | 370,0 |
| `climate_zone` | mixed | 6.718 | 367 | 19,233% | 95,570 | 403,0 | 336,0 |
| `climate_zone` | cold | 6.451 | 357 | 18,469% | 95,040 | 437,5 | 410,0 |
| `climate_zone` | temperate | 6.036 | 342 | 17,281% | 95,310 | 418,0 | 371,0 |
| `climate_zone` | Missing | 1.950 | 1.188 | 5,583% | 95,150 | 412,0 | 359,0 |
| `bms_firmware_version` | BMS_v3.0 | 6.783 | 1.785 | 19,419% | 95,150 | 417,0 | 374,0 |
| `bms_firmware_version` | BMS_v2.3 | 6.687 | 1.780 | 19,145% | 95,260 | 420,0 | 371,0 |
| `bms_firmware_version` | BMS_v3.5 | 6.676 | 1.772 | 19,113% | 95,380 | 414,0 | 365,0 |
| `bms_firmware_version` | BMS_v2.1 | 6.609 | 1.757 | 18,921% | 95,060 | 430,5 | 382,0 |
| `bms_firmware_version` | BMS_v3.2 | 6.593 | 1.773 | 18,875% | 95,310 | 409,0 | 364,0 |
| `bms_firmware_version` | Missing | 1.581 | 1.072 | 4,526% | 95,230 | 424,0 | 378,5 |
| `recall_flag` | 0.0 | 32.565 | 1.820 | 93,232% | 95,230 | 418,0 | 372,0 |
| `recall_flag` | 1.0 | 519 | 460 | 1,486% | 95,370 | 416,0 | 368,0 |
| `recall_flag` | Missing | 1.845 | 1.147 | 5,282% | 95,270 | 414,0 | 373,0 |

Terdapat beberapa temuan penting dari struktur kategori di atas. Pertama, pada `chemistry`, jenis baterai NMC (31,160%) dan LFP (28,071%) mendominasi proporsi observasi, diikuti oleh NCA (19,998%) dan LMO (13,885%). Kedua, jumlah baterai unik pada setiap versi `bms_firmware_version` berkisar antara 1.757 hingga 1.785 dari total 1.820 baterai *train*. Angka ini membuktikan bahwa versi firmware BMS berubah-ubah antarobservasi di dalam baterai yang sama, sehingga tidak boleh diperlakukan sebagai atribut statis baterai. Hal serupa berlaku pada `recall_flag`, di mana kategori `1.0` sangat langka (hanya 519 baris atau 1,486% observasi yang tersebar di 460 baterai), sementara seluruh 1.820 baterai pernah memiliki observasi dengan `recall_flag = 0.0`. Ketiga, keberadaan kategori `Missing` yang tersebar di ribuan baterai berbeda (misalnya 1.345 baterai pernah mengalami nilai kosong pada kolom `chemistry` di salah satu baris observasinya) menegaskan bahwa nilai kosong diinjeksikan pada tingkat baris observasi, bukan pada tingkat unit baterai secara keseluruhan.

---

## BAB IV: Analisis Hubungan Antarvariabel, Interaksi Kimia-Umur, dan Redundansi Fitur

### 4.1 Perbandingan SOH Berdasarkan Kimia Baterai dan Umur Pakai

Membandingkan median SOH antarkimia baterai secara agregat berisiko menyesatkan apabila distribusi umur baterai di setiap kelompok kimia tidak seimbang. Untuk memperoleh perbandingan yang lebih adil pada [01_eda.ipynb](../../notebooks/01_eda.ipynb), dibentuk label bantu khusus analisis bernama `chemistry_group` yang mengambil nilai kimia tunggal yang konsisten dari observasi non-kosong milik masing-masing baterai di dalam *train* (tanpa mengubah atau mengimputasi kolom `chemistry` asli untuk pemodelan). Selanjutnya, umur baterai dikelompokkan ke dalam enam pita tahunan (`<=1 year`, `1-2 years`, `2-3 years`, `3-4 years`, `4-5 years`, dan `>5 years`). Di dalam setiap sel kombinasi kimia dan pita umur, nilai SOH dirata-ratakan terlebih dahulu per baterai, kemudian diambil rerata lintas baterai dengan bobot yang sama untuk setiap baterai (*battery-balanced mean SOH*), dilengkapi interval kepercayaan *percentile bootstrap* 95% dari 1.000 kali pengambilan sampel ulang tingkat baterai (`seed = 42`).

![Perbandingan SOH berdasarkan Kimia Baterai dan Kurva Umur Berbobot Sama per Baterai](04_chemistry_and_age.png)

Sebagai ilustrasi kuantitatif pada rentang umur yang sama, tabel berikut menampilkan perbandingan keempat jenis kimia baterai pada pita umur **1-2 tahun**:

| Jenis Kimia (`chemistry`) | Jumlah Baris (`rows`) | Jumlah Baterai (`batteries`) | Rerata SOH Berbobot Sama per Baterai (%) | Batas Bawah CI 95% (`ci95_low`) | Batas Atas CI 95% (`ci95_high`) |
|---|---:|---:|---:|---:|---:|
| **LFP** | 2.439 | 469 | **96,237%** | 96,119% | 96,348% |
| **NMC** | 2.718 | 501 | **92,845%** | 92,634% | 93,038% |
| **NCA** | 1.665 | 344 | **91,817%** | 91,513% | 92,115% |
| **LMO** | 1.168 | 226 | **89,270%** | 88,781% | 89,709% |

Data di atas memperlihatkan bahwa bahkan setelah dikondisikan pada rentang umur 1 hingga 2 tahun dengan bobot setara per baterai, urutan rata-rata kesehatan baterai tetap berbeda secara konsisten: LFP memiliki SOH tertinggi (96,237%), disusul oleh NMC (92,845%), NCA (91,817%), dan LMO (89,270%). Seiring bertambahnya kelompok umur hingga di atas 5 tahun, seluruh kimia menunjukkan tren penurunan SOH dengan kemiringan yang berbeda-beda.

![Jumlah Baterai Pendukung per Sel Kimia dan Rentang Umur](05_chemistry_age_support.png)

Pemeriksaan dukungan sampel (*sample support*) pada Gambar 05 di atas menunjukkan bahwa hampir seluruh sel kombinasi kimia dan umur didukung oleh minimal 30 unit baterai, kecuali satu sel yakni **LMO pada kelompok umur `>5 years`** yang hanya memiliki **51 baris observasi dari 22 baterai** (ditandai dengan simbol `x` pada kurva). Temuan ini menggarisbawahi dua implikasi utama: pertama, jenis kimia dan umur baterai berinteraksi dalam menentukan laju penurunan SOH di dalam simulator sehingga keduanya wajib dipertahankan sebagai fitur input bagi MLP; kedua, perbedaan kurva ini merupakan pola di dalam simulator yang belum mengontrol faktor pemakaian lain secara serentak, sehingga tidak boleh diklaim sebagai bukti kausalitas fisik armada kendaraan di dunia nyata.

### 4.2 Asosiasi Fitur Numerik terhadap SOH dan Dekomposisi Variasi

Untuk mengidentifikasi fitur numerik mana yang memiliki hubungan paling kuat dengan `state_of_health_pct`, dihitung koefisien korelasi linear Pearson dan korelasi peringkat monoton Spearman menggunakan pasangan baris yang lengkap (*pairwise-complete observations*) tanpa imputasi, dengan jumlah pasangan berkisar antara 30.368 hingga 33.825 baris. Selain itu, karena adanya struktur pengelompokan baterai, korelasi Pearson didekomposisi menjadi tiga sudut pandang: korelasi gabungan seluruh baris (*pooled*), korelasi antar-rerata baterai (*between-battery means*), dan korelasi di dalam baterai setelah nilai setiap observasi dikurangi rerata baterainya masing-masing (*within-battery centered*).

![Matriks Korelasi Fitur Baseline terhadap SOH](06_baseline_correlations.png)

![Asosiasi Fitur terhadap Target dan Dekomposisi Korelasi Pooled, Between-Battery, serta Within-Battery](07_target_associations_and_dependence.png)

Tabel berikut menyajikan peringkat seluruh 20 variabel (15 fitur numerik utama, 1 atribut biner, dan 4 *proxy* sensor) berdasarkan kekuatan korelasi Spearman absolut terhadap SOH pada data *train*:

| Peringkat | Nama Fitur | Peran | Baris Berpasangan | Pearson (*Pooled*) | Spearman (*Pooled*) | Pearson (*Between-Battery*) | Pearson (*Within-Battery*) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `cycle_count` | `primary` | 33.337 | -0,873 | **-0,907** | -0,829 | -0,922 |
| 2 | `internal_resistance_mohm` | `proxy_not_baseline` | 33.465 | -0,928 | **-0,896** | -0,916 | -0,942 |
| 3 | `total_energy_throughput_kwh` | `primary` | 32.567 | -0,822 | **-0,886** | -0,767 | -0,892 |
| 4 | `battery_age_days` | `primary` | 32.814 | -0,809 | **-0,881** | -0,731 | -0,898 |
| 5 | `service_count` | `primary` | 33.825 | -0,564 | **-0,593** | -0,685 | -0,542 |
| 6 | `charge_cycles_per_month` | `primary` | 33.503 | -0,275 | **-0,244** | -0,406 | 0,010 |
| 7 | `avg_daily_distance_km` | `primary` | 33.444 | -0,177 | **-0,169** | -0,377 | -0,004 |
| 8 | `avg_ambient_temp_c` | `primary` | 32.659 | 0,024 | 0,056 | 0,006 | 0,052 |
| 9 | `peak_temp_during_fast_charge_c` | `primary` | 33.439 | 0,022 | 0,053 | 0,004 | 0,042 |
| 10 | `voltage_avg_v` | `proxy_not_baseline` | 33.002 | 0,048 | 0,023 | 0,078 | 0,008 |
| 11 | `avg_depth_of_discharge_pct` | `primary` | 32.933 | -0,033 | -0,021 | -0,049 | -0,020 |
| 12 | `avg_state_of_charge_pct` | `primary` | 32.778 | 0,027 | 0,019 | 0,048 | 0,008 |
| 13 | `recall_flag` | `primary` | 33.084 | 0,004 | 0,006 | 0,024 | 0,000 |
| 14 | `current_avg_a` | `proxy_not_baseline` | 33.157 | -0,004 | -0,005 | -0,015 | -0,000 |
| 15 | `charge_time_avg_min` | `proxy_not_baseline` | 33.516 | -0,002 | 0,003 | -0,007 | 0,004 |
| 16 | `fast_charge_ratio` | `primary` | 32.563 | -0,002 | -0,003 | -0,005 | -0,011 |
| 17 | `altitude_m` | `primary` | 33.297 | -0,005 | 0,002 | 0,037 | -0,016 |
| 18 | `avg_charging_power_kw` | `primary` | 33.416 | 0,001 | 0,002 | -0,005 | -0,001 |
| 19 | `pack_capacity_kwh` | `primary` | 33.345 | 0,038 | -0,001 | 0,044 | NaN (konstan) |
| 20 | `humidity_pct` | `primary` | 32.993 | 0,002 | -0,000 | -0,006 | 0,009 |

Di antara fitur-fitur utama (*primary features*), tiga variabel akumulasi pemakaian menunjukkan asosiasi negatif yang sangat kuat terhadap SOH, yaitu `cycle_count` (Spearman `-0,907`, Pearson `-0,873`), `total_energy_throughput_kwh` (Spearman `-0,886`, Pearson `-0,822`), dan `battery_age_days` (Spearman `-0,881`, Pearson `-0,809`), diikuti oleh `service_count` (Spearman `-0,593`). Fakta bahwa nilai mutlak korelasi Spearman pada ketiga variabel utama tersebut secara konsisten lebih tinggi dibandingkan korelasi Pearson menunjukkan bahwa hubungan penurunan SOH terhadap akumulasi siklus, energi, dan umur bersifat monoton namun memiliki komponen kelengkungan (*nonlinearity*).

Dekomposisi korelasi pada panel kanan Gambar 07 juga memberikan wawasan menarik. Untuk `charge_cycles_per_month` dan `avg_daily_distance_km`, korelasi antar-rerata baterai cukup terasa (`-0,406` dan `-0,377`), tetapi korelasi di dalam baterai yang sama (*within-battery centered*) mendekati nol (`0,010` dan `-0,004`). Artinya, perbedaan intensitas pemakaian bulanan dan jarak tempuh harian terutama membedakan cepat-lambatnya degradasi antara satu unit baterai dengan unit baterai lainnya, bukan fluktuasi dari waktu ke waktu di dalam satu baterai. Sebaliknya, `pack_capacity_kwh` memiliki korelasi *within-battery* `NaN` karena kapasitas nominal selalu konstan untuk setiap unit baterai. Sementara itu, kolom `internal_resistance_mohm` memiliki korelasi linear Pearson yang sangat tinggi terhadap SOH (`-0,928` secara *pooled* dan `-0,942` di dalam baterai), yang semakin memperkuat temuan audit bahwa resistansi internal di dalam dataset sintetis ini dibangkitkan langsung dari fungsi SOH.

### 4.3 Multikolinearitas dan Redundansi Antarfitur Utama

Tingginya asosiasi `cycle_count`, `total_energy_throughput_kwh`, dan `battery_age_days` terhadap SOH diiringi oleh korelasi silang yang sangat kuat di antara ketiganya. Dengan menggunakan ambang eksploratif `|Spearman| >= 0,70`, teridentifikasi enam pasangan fitur utama yang memiliki tingkat kolinearitas atau redundansi tinggi pada data *train*:

| Fitur Pertama (`feature_a`) | Fitur Kedua (`feature_b`) | Korelasi Spearman | Jumlah Baris Berpasangan |
|---|---|---:|---:|
| `cycle_count` | `total_energy_throughput_kwh` | **0,984** | 31.064 |
| `avg_ambient_temp_c` | `peak_temp_during_fast_charge_c` | **0,920** | 31.255 |
| `battery_age_days` | `cycle_count` | **0,903** | 31.304 |
| `battery_age_days` | `total_energy_throughput_kwh` | **0,890** | 30.582 |
| `avg_charging_power_kw` | `fast_charge_ratio` | **0,841** | 31.147 |
| `avg_depth_of_discharge_pct` | `avg_state_of_charge_pct` | **-0,767** | 30.917 |

Pasangan `cycle_count` dan `total_energy_throughput_kwh` memiliki korelasi peringkat mencapai `0,984`, karena total energi yang mengalir keluar-masuk baterai hampir sepenuhnya merupakan hasil kali antara jumlah siklus pengisian dengan kapasitas paket baterai. Demikian pula, `peak_temp_during_fast_charge_c` berkorelasi sebesar `0,920` dengan `avg_ambient_temp_c`. Ketiga indikator akumulasi (`battery_age_days`, `cycle_count`, dan `total_energy_throughput_kwh`) bukan merupakan tiga bukti independen, melainkan tiga proyeksi dari proses penuaan yang sama. Temuan redundansi ini menjadi landasan ilmiah untuk merancang resep seleksi fitur `compact_log_sensor_caps` pada tahap *preprocessing*, yang menguji penghapusan `total_energy_throughput_kwh` dan `peak_temp_during_fast_charge_c` (beserta indikator *missing*-nya, sehingga menyusutkan dimensi input dari 67 menjadi 63 neuron) untuk dievaluasi pada himpunan *validation*.

### 4.4 Pola Nonlinearitas Lintas Kimia dan Analisis Bersyarat (*Conditioning*)

Visualisasi diagram pencar (*scatter plot*) pada sampel deterministik 6.000 observasi *train* yang diwarnai berdasarkan jenis kimia baterai memperjelas mengapa model regresi linear tunggal tidak memadai untuk studi kasus ini:

![Hubungan Fitur Numerik Utama terhadap SOH yang Dikelompokkan Berdasarkan Kimia Baterai](08_relationships_by_chemistry.png)

Pada grafik `cycle_count`, `battery_age_days`, dan `total_energy_throughput_kwh` terhadap SOH, terlihat pemisahan pita lintasan yang jelas antarkimia baterai. Untuk jumlah siklus atau umur pakai yang sama, titik-titik observasi baterai LFP berada pada pita SOH yang paling atas dengan kemiringan penurunan yang lebih landai, diikuti oleh NMC dan NCA di bagian tengah, serta LMO yang menurun paling curam menuju batas SOH 55%. Kemiringan dan sebaran yang bergantung pada interaksi antara fitur numerik dan fitur kategorikal ini memberikan justifikasi kuat bagi penggunaan arsitektur *Multilayer Perceptron* dengan lapisan tersembunyi (*hidden layer*) nonlinear.

Lebih jauh, korelasi *pooled* yang tampak mendekati nol pada fitur seperti `fast_charge_ratio` (Spearman `-0,003`) tidak serta-merta membuktikan bahwa rasio pengisian cepat tidak berpengaruh di dalam simulator, sebab efeknya dapat tertutup oleh variasi umur dan kimia baterai (*confounding*). Untuk memisahkan variasi umur dari variasi pengisian cepat, dilakukan analisis pengondisian (*conditioning*) pada matriks pita umur dan pita `fast_charge_ratio` (hanya menampilkan sel dengan minimal 100 baris dan 30 baterai), serta penghitungan korelasi Spearman terstratifikasi di dalam 23 sel kombinasi kimia dan umur yang memenuhi syarat dukungan sampel:

![Kondisioning Median SOH pada Matriks Kelompok Umur dan Rasio Fast Charge](09_age_fastcharge_conditioning.png)

| Fitur yang Distratifikasi (23 Sel Kimia x Umur) | Median Spearman (`median_rho`) | Spearman Minimum (`min_rho`) | Spearman Maksimum (`max_rho`) |
|---|---:|---:|---:|
| `charge_cycles_per_month` | **-0,770** | -0,828 | -0,249 |
| `service_count` | -0,090 | -0,208 | 0,128 |
| `avg_depth_of_discharge_pct` | -0,076 | -0,284 | -0,007 |
| `fast_charge_ratio` | -0,026 | -0,263 | 0,159 |
| `avg_ambient_temp_c` | 0,057 | -0,216 | 0,185 |

Tabel asosiasi terstratifikasi di atas mengungkap fenomena penting yang tersembunyi pada korelasi gabungan: ketika umur dan kimia baterai dikendalikan di dalam sel yang sama, `charge_cycles_per_month` menunjukkan korelasi negatif yang sangat kuat terhadap SOH dengan median Spearman mencapai `-0,770` (berkisar antara `-0,828` hingga `-0,249`), padahal korelasi *pooled*-nya hanya `-0,244`. Hal ini terjadi karena pada umur kalender yang sama, baterai yang mengalami frekuensi siklus bulanan lebih tinggi otomatis telah mengakumulasi total siklus yang jauh lebih banyak. Sementara itu, `fast_charge_ratio` dan `avg_depth_of_discharge_pct` menunjukkan median korelasi negatif tipis di dalam sel (`-0,026` dan `-0,076`), dengan efek negatif yang lebih terlihat pada baterai berumur tua (misalnya Spearman `-0,263` pada NCA `>5 years` dan `-0,173` pada LFP `>5 years`).

![Perbandingan Median SOH, Dukungan Sampel, dan Median Umur Lintas Kategori](10_category_comparisons.png)

Pengaruh variabel perancu (*confounder*) umur juga terlihat jelas ketika membandingkan kategori produsen (`manufacturer`), model kendaraan (`vehicle_model`), zona iklim (`climate_zone`), dan versi firmware (`bms_firmware_version`) pada Gambar 10. Selisih median SOH antarkategori tersebut relatif sempit (seluruhnya berada di kisaran 94,37% hingga 95,91%) dan berbanding terbalik dengan median umur sampel di setiap kategori. Sebagai contoh, produsen `CellTech` memiliki median SOH terendah (94,370%) karena kebetulan memiliki median umur baterai paling tua di dalam data *train* (479,0 hari dan 456,0 siklus), sedangkan model `Hatch-E3` memiliki median SOH tertinggi (95,910%) karena memiliki median umur paling muda (379,0 hari dan 327,0 siklus). Oleh karena itu, tidak ada fitur kategorikal maupun fitur berkorelasi marginal rendah yang dibuang secara prematur hanya berdasarkan pengamatan univariat pada tahap EDA.

---

## BAB V: Diagnostik Data Hilang (*Missingness*), *Outlier*, Dinamika Temporal, dan Target Alternatif

### 5.1 Pola Nilai Hilang (*Missing Values*) dan Mekanisme Struktural Tanggal Servis

Sebagaimana teridentifikasi pada audit skema, 29 dari 35 kolom di dalam dataset memiliki nilai hilang. Untuk memahami apakah ketiadaan nilai pada suatu fitur berkaitan dengan kondisi kesehatan baterai atau saling bergantung dengan ketiadaan nilai pada fitur lain, dilakukan tiga diagnostik pada data *train*: distribusi jumlah kolom yang hilang per baris observasi, perbandingan median SOH antara baris yang hilang (*missing*) dan baris yang terisi (*observed*), serta korelasi koefisien *phi* antarindikator *missing* dari seluruh pasangan kolom.

![Diagnostik Missing Value pada Data Train](11_missingness_diagnostics.png)

Pada ke-21 fitur utama (*primary features*), persentase baris yang hilang di dalam data *train* bergerak di rentang yang cukup seragam, yakni dari `3,161%` (`service_count`, 1.104 baris) hingga `6,908%` (`vehicle_model`, 2.413 baris) dan `6,885%` (`chemistry`, 2.405 baris). Selisih median SOH antara baris yang nilainya hilang dan baris yang nilainya teramati (`median_soh_missing - median_soh_observed`) pada seluruh 21 fitur utama tersebut sangat kecil, yaitu hanya berkisar antara **-0,240 hingga +0,280 poin persentase** (dengan median umur baterai yang juga berdekatan di kisaran 399,5 hingga 446,5 hari). Selain itu, korelasi linear antarindikator *missing* di antara seluruh pasangan kolom menunjukkan nilai mutlak maksimum `|phi|` hanya sebesar **0,017** (terjadi pada pasangan `avg_depth_of_discharge_pct` dengan `last_service_date` serta `climate_zone` dengan `total_energy_throughput_kwh`). Pola ini konsisten dengan mekanisme injeksi nilai hilang secara acak dan independen per fitur oleh simulator, meskipun secara statistik tetap tidak dapat diklaim sebagai bukti mutlak *Missing Completely At Random* (MCAR).

![Konteks Struktural Nilai Hilang pada Tanggal Servis Terakhir](12_service_missingness_context.png)

Pengecualian yang sangat mencolok terjadi pada kolom `last_service_date`, yang kosong pada 13.198 baris (37,785%) di dalam himpunan *train* (atau 18.831 baris / 37,662% pada dataset penuh). Sebagaimana ditunjukkan pada Gambar 12 dan tabel rincian berikut, ketiadaan tanggal servis sama sekali bukan merupakan kerusakan data acak:

| Pemeriksaan Konteks `last_service_date` | Nilai Terukur |
|---|---:|
| Baris `last_service_date` Kosong dengan `service_count = 0` (Dataset Penuh) | 18.205 baris |
| Baris `last_service_date` Kosong dengan `service_count > 0` (Dataset Penuh) | **0 baris** |
| Baris `last_service_date` Kosong dengan `service_count` ikut Kosong (Dataset Penuh) | 626 baris |
| Median SOH saat `last_service_date` Kosong (*Train*) | **97,790%** |
| Median SOH saat `last_service_date` Terisi (*Train*) | **92,700%** |
| Selisih Median SOH (*Missing* dikurangi *Observed* pada *Train*) | **+5,090 poin persentase** |
| Median Umur Baterai saat `last_service_date` Kosong (*Train*) | **174,0 hari** |
| Median Umur Baterai saat `last_service_date` Terisi (*Train*) | **662,0 hari** |

Dari 18.831 baris yang tidak memiliki tanggal servis terakhir, 18.205 di antaranya memiliki `service_count = 0`, 626 sisanya mengalami *missing value* pada `service_count`, dan tepat 0 baris yang memiliki `service_count > 0`. Di dalam data *train*, baterai yang tanggal servisnya kosong memiliki median umur yang jauh lebih muda (174,0 hari) dan median SOH yang jauh lebih tinggi (97,790%) dibandingkan baterai yang memiliki catatan tanggal servis (median umur 662,0 hari dan median SOH 92,700%). Dengan demikian, kekosongan `last_service_date` bersifat struktural karena baterai baru memang belum pernah menjalani servis perawatan, bukan karena tindakan servis merusak kesehatan baterai.

### 5.2 Deteksi Kandidat *Outlier* dan Analisis Sensitivitas Korelasi

Untuk memetakan keberadaan nilai ekstrem pada distribusi numerik di data *train*, diterapkan aturan pagar kuartil `1,5 x IQR` (*Interquartile Range*), di mana batas bawah didefinisikan sebagai `Q1 - 1,5 x IQR` dan batas atas sebagai `Q3 + 1,5 x IQR`. Penting ditegaskan bahwa aturan ini hanya digunakan sebagai instrumen diagnostik untuk menandai kandidat pencilan (*flagged candidates*), bukan aturan otomatis untuk membuang baris data.

![Persentase Kandidat Outlier berdasarkan 1,5 x IQR dan Distribusi Ekor Panjang](13_outliers_and_tail.png)

Tabel berikut merangkum batas pagar IQR pada data *train*, jumlah dan persentase baris yang tertandai di luar pagar, serta hasil analisis sensitivitas korelasi Spearman terhadap SOH ketika seluruh baris dipakai dibandingkan ketika hanya baris di dalam pagar IQR yang dihitung:

| Nama Fitur | Batas Bawah IQR | Batas Atas IQR | Baris Tertandai | % Tertandai | Spearman (Semua Baris) | Spearman (Dalam Pagar IQR) |
|---|---:|---:|---:|---:|---:|---:|
| `charge_time_avg_min` (*proxy*) | -76,300 | 288,500 | 2.252 | **6,719%** | 0,003 | 0,012 |
| `total_energy_throughput_kwh` | -57.895,150 | 119.884,450 | 1.837 | **5,641%** | -0,886 | -0,870 |
| `internal_resistance_mohm` (*proxy*) | 1,515 | 90,275 | 1.852 | **5,534%** | -0,896 | -0,877 |
| `cycle_count` | -860,000 | 1.796,000 | 1.684 | **5,051%** | -0,907 | -0,895 |
| `state_of_health_pct` (target) | 78,125 | 110,005 | 1.729 | **4,950%** | - | - |
| `altitude_m` | -409,000 | 911,000 | 1.574 | **4,727%** | 0,002 | 0,004 |
| `avg_daily_distance_km` | -43,950 | 133,250 | 1.523 | **4,554%** | -0,169 | -0,156 |
| `service_count` | -3,000 | 5,000 | 1.043 | **3,084%** | -0,593 | -0,568 |
| `avg_charging_power_kw` | -15,200 | 76,000 | 787 | 2,355% | 0,002 | 0,000 |
| `recall_flag` (biner) | 0,000 | 0,000 | 519 | 1,569% | 0,006 | `NaN` (hanya nilai 0) |
| `current_avg_a` (*proxy*) | 87,350 | 187,750 | 474 | 1,430% | -0,005 | -0,003 |
| `battery_age_days` | -835,000 | 1.837,000 | 445 | 1,356% | -0,881 | -0,877 |
| `charge_cycles_per_month` | -7,900 | 47,300 | 331 | 0,988% | -0,244 | -0,239 |
| `fast_charge_ratio` | -0,205 | 0,771 | 216 | 0,663% | -0,003 | -0,003 |
| `avg_ambient_temp_c` | -11,900 | 54,500 | 156 | 0,478% | 0,056 | 0,054 |
| `peak_temp_during_fast_charge_c` | -8,075 | 61,725 | 154 | 0,461% | 0,053 | 0,053 |
| `avg_depth_of_discharge_pct` | 21,400 | 108,600 | 130 | 0,395% | -0,021 | -0,021 |
| `avg_state_of_charge_pct` | 40,100 | 95,300 | 121 | 0,369% | 0,019 | 0,018 |
| `humidity_pct` | 6,250 | 103,450 | 111 | 0,336% | -0,000 | 0,000 |
| `voltage_avg_v` (*proxy*) | 328,537 | 399,638 | 43 | 0,130% | 0,023 | 0,024 |
| `pack_capacity_kwh` | 32,500 | 112,500 | 0 | 0,000% | -0,001 | -0,001 |

Hasil diagnostik di atas memperlihatkan tiga fakta krusial. Pertama, apabila aturan `1,5 x IQR` diterapkan secara membabi buta untuk membuang baris, maka **1.729 baris (4,950%)** yang memiliki `state_of_health_pct < 78,125%` akan terhapus sebagai *outlier*, yang berarti menghilangkan hampir seluruh sampel baterai terdegradasi di bawah ambang *end-of-life* 80%. Hal serupa terjadi pada `recall_flag`, di mana kuartil ke-25 dan ke-75 sama-sama bernilai `0,000`, sehingga pagar IQR membuang seluruh 519 baris yang bernilai `1,000` dan membuat variasi fitur menjadi nol (`NaN`). Kedua, pada fitur-fitur bersensor suhu dan daya seperti `avg_ambient_temp_c` (maksimum `83,715` derajat C), `peak_temp_during_fast_charge_c` (maksimum `89,820` derajat C), dan `avg_charging_power_kw` (maksimum `147,772` kW), terdapat ekor ekstrem yang sesuai dengan klaim dokumentasi mengenai adanya injeksi *sensor outliers* oleh simulator. Ketiga, uji sensitivitas membuktikan bahwa korelasi Spearman pada fitur-fitur utama hampir tidak berubah ketika nilai di luar pagar IQR disaring (`cycle_count` dari `-0,907` menjadi `-0,895`, `battery_age_days` dari `-0,881` menjadi `-0,877`, dan `total_energy_throughput_kwh` dari `-0,886` menjadi `-0,870`). Artinya, hubungan sinyal utama terhadap SOH tidak digerakkan oleh pencilan. Sebagai langkah tindak lanjut yang terukur pada tahap *preprocessing*, tidak ada baris yang dihapus, melainkan diuji resep alternatif yang melakukan pemotongan (*capping*) pada persentil 0,5% dan 99,5% khusus untuk tiga fitur bersensor ekstrem (`avg_ambient_temp_c`, `peak_temp_during_fast_charge_c`, dan `avg_charging_power_kw`) dikombinasikan dengan transformasi `log1p` pada lima fitur berekor panjang.

### 5.3 Lintasan SOH Individu dan Evaluasi Kualitas Urutan Waktu

Untuk memeriksa perilaku longitudinal pada tingkat unit baterai di dalam himpunan *train*, diambil sampel deterministik berupa tiga `battery_id` pertama secara alfabetis dari setiap jenis kimia yang memiliki minimal 10 observasi bertanggal valid (tanpa memilih berdasarkan nilai SOH).

![Contoh Deterministik Lintasan SOH Seiring Tanggal Observasi per Kimia Baterai](14_battery_trajectories.png)

Kurva lintasan pada Gambar 14 memperlihatkan bahwa meskipun tren jangka panjang SOH menurun seiring berjalannya waktu kalender, lintasan individu tiap baterai tidak bergerak turun secara mulus (*non-monotonic*), melainkan sering mengalami lompatan naik dan osilasi tajam antarobservasi yang berdekatan. Guna mengukur fenomena ini secara kuantitatif pada seluruh populasi *train*, dilakukan analisis terhadap **31.528 pasangan observasi berurutan** dari baterai yang sama yang memiliki tanggal valid dan menaik (`gap_days > 0`), dengan pembulatan perubahan SOH (`delta_soh_pp`) ke presisi 0,01 poin persentase:

![Diagnostik Kualitas Urutan Waktu: Distribusi Selang Hari, Perubahan SOH, dan Arah Perubahan](15_temporal_quality.png)

| Parameter Pasangan Berurutan (*Train*) | Jumlah Pasangan (`count`) | Mean | Std | Min | P1 | Median (P50) | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Selisih Hari (`gap_days`) | 31.528 | 57,753 | 72,089 | 1,000 | 1,000 | 32,000 | 345,460 | 853,000 |
| Perubahan SOH (`delta_soh_pp`, poin %) | 31.528 | -0,700 | 1,626 | -22,260 | -5,737 | -0,530 | 2,490 | 5,550 |
| Perubahan Siklus (`delta_cycles`) | 28.749 | 59,361 | 112,174 | -634,000 | -150,520 | 29,000 | 487,000 | 1.463,000 |
| Perubahan Energi (`delta_energy_kwh`) | 27.395 | 4.011,997 | 8.554,773 | -64.595,000 | -13.846,630 | 1.924,500 | 35.411,764 | 131.941,900 |

Apabila perubahan SOH pada ke-31.528 pasangan berurutan tersebut diklasifikasikan berdasarkan arah pergerakannya, diperoleh proporsi sebagai berikut:

- **Menurun (`down > 0,01 pp`):** 20.498 pasangan (**65,015%**).
- **Stabil (`stable +/- 0,01 pp`):** 670 pasangan (**2,125%**).
- **Meningkat (`up > 0,01 pp`):** 10.360 pasangan (**32,860%**), dengan kenaikan mencapai `+2,490` poin persentase pada persentil ke-99 dan maksimum `+5,550` poin persentase.

Fakta bahwa hampir sepertiga (32,860%) dari pasangan pengamatan berurutan menunjukkan kenaikan SOH yang signifikan, ditambah dengan selang waktu pengamatan yang sangat tidak beraturan (berkisar dari 1 hari hingga 853 hari, median 32 hari, simpangan baku 72,089 hari) serta perubahan siklus dan energi yang sering bernilai negatif (hingga `-634` siklus dan `-64.595` kWh), mengonfirmasi bahwa fluktuasi naik tersebut merupakan *noise* observasional dari simulator dan bukan proses pemulihan elektrokimia baterai secara fisik. Temuan ini menutup kemungkinan pemodelan laju degradasi harian maupun *forecasting* berbasis *timesteps*, sehingga mempertahankan pendekatan regresi tabular per observasi adalah keputusan yang paling tepat.

### 5.4 Eksplorasi Kontekstual Target Alternatif

Walaupun ketiga kolom target alternatif (`remaining_useful_life_cycles`, `failure_risk_label`, dan `degradation_category`) telah dikeluarkan dari daftar fitur input agar tidak terjadi kebocoran informasi kesehatan baterai, ketiganya tetap dipetakan di dalam [01_eda.ipynb](../../notebooks/01_eda.ipynb) semata-mata untuk memahami struktur pelabelan di dalam simulator:

![Profil Target Alternatif pada Data Train: RUL, Failure Risk Label, dan Degradation Category](17_alternative_targets.png)

| Fitur Target Alternatif | Kategori | Jumlah Baris (*Train*) | Jumlah Baterai | Persentase Baris (%) | Median SOH (%) |
|---|---|---:|---:|---:|---:|
| `failure_risk_label` | `Low` | 29.781 | 1.820 | 85,262% | 96,220% |
| `failure_risk_label` | `Medium` | 3.087 | 758 | 8,838% | 83,520% |
| `failure_risk_label` | `High` | 2.061 | 403 | 5,901% | 74,520% |
| `degradation_category` | `Normal` | 32.179 | 1.820 | 92,127% | 95,770% |
| `degradation_category` | `Accelerated` | 2.225 | 494 | 6,370% | 77,760% |
| `degradation_category` | `Critical` | 525 | 128 | 1,503% | 65,270% |

Sebaran di atas memperlihatkan korespondensi langsung antara label klasifikasi alternatif dengan rentang nilai SOH pada data *train*. Sebagai contoh, kategori `Critical` pada `degradation_category` berjumlah tepat 525 baris (1,503%), identik dengan jumlah baris pada pita SOH `< 70%` di Tabel 3.1, sedangkan gabungan kategori `Accelerated` (2.225 baris, median SOH 77,760%) dan `Critical` (525 baris, median SOH 65,270%) merefleksikan ekor baterai yang mengalami penurunan kapasitas menuju dan melewati batas *end-of-life* 80%. Hal ini menegaskan bahwa ketiga target alternatif diturunkan dari status kesehatan yang sama dengan SOH dan tidak boleh digunakan sebagai input prediktor.

---

## BAB VI: Sintesis Temuan dan Implikasi terhadap Perancangan *Preprocessing* serta Arsitektur MLP

### 6.1 Keputusan Desain *Preprocessing* dan Rekayasa Fitur

Seluruh bukti kuantitatif yang diperoleh dari tahapan audit data ([00_data_audit.ipynb](../../notebooks/00_data_audit.ipynb)) dan *Exploratory Data Analysis* ([01_eda.ipynb](../../notebooks/01_eda.ipynb)) menjadi dasar langsung bagi perancangan modul [preprocessing.py](../../src/ev_battery/preprocessing.py) dan konfigurasi eksperimen MLP pada [METHODOLOGY.md](../METHODOLOGY.md). Hubungan antara setiap temuan empiris EDA dan keputusan teknis pada tahapan berikutnya dirangkum dalam matriks sintesis berikut:

| Aspek Temuan Audit & EDA | Bukti Empiris Utama pada Data *Train* | Implikasi Teknis pada *Preprocessing* & Pemodelan MLP | Batasan Interpretasi |
|---|---|---|---|
| **Pencegahan Kebocoran Target (*Leakage*)** | `capacity_fade_pct + SOH = 100` (galat 0,0); `internal_resistance_mohm` berkorelasi Pearson `-0,928` terhadap SOH. | Mengeluarkan `capacity_fade_pct`, 3 target alternatif, 2 kolom ID, 3 tanggal mentah, dan 4 *proxy* sensor; menyisakan 21 fitur utama. | Penahanan sensor bersifat khusus untuk simulator ini, bukan larangan penggunaan sensor riil pada BMS fisik. |
| **Penanganan Nilai Hilang (*Missing Values*)** | Nilai hilang tersebar di 21 fitur utama (3,16% - 6,91%); selisih median SOH baris hilang vs terisi hanya -0,24 s.d. +0,28 poin %. | Mengimputasi fitur numerik dengan median *train* ditambah indikator biner *missing* per fitur; mengodekan kategori kosong sebagai `__MISSING__`. | Selisih median yang kecil tidak membuktikan kondisi MCAR secara teoritis. |
| **Pengodean Variabel Kategorikal** | 5 fitur kategorikal dengan jumlah kategori unik 4 hingga 8 di *train*, ditambah kategori `__MISSING__`. | Menerapkan *one-hot encoding* tanpa urutan semu; menghasilkan total **67 neuron input** pada resep *baseline* (dan **63 neuron input** pada resep *compact*). | Kategori baru di luar *train* dipetakan ke vektor nol agar dimensi input jaringan saraf tetap stabil. |
| **Normalisasi Skala Fitur** | Rentang nilai berbeda ekstrem, dari rasio `0 - 0,937` hingga energi `402.419,9 kWh`. | Menerapkan `StandardScaler` pada fitur numerik sebagai *baseline* dan membandingkannya terhadap `MinMaxScaler` (`baseline_minmax`); target distandardisasi terpisah. | Seluruh parameter *scaler* (mean, std, min, max) di-*fit* hanya dari data *train*. |
| **Penanganan Ekor Panjang & *Outlier*** | 5 fitur utama memiliki *skewness* tinggi (1,62 - 2,18); aturan `1,5 x IQR` justru menandai 4,95% baterai ber-SOH rendah sebagai *outlier*. | Tidak menghapus baris berdasarkan IQR; menguji resep `log_sensor_caps` dengan transformasi `log1p` pada 5 fitur miring dan *capping* kuantil 0,5% / 99,5% pada 3 fitur suhu/daya. | Batas IQR hanya penanda distribusi ekstrem, bukan bukti adanya kesalahan pencatatan data. |
| **Seleksi Fitur Redundan** | Korelasi Spearman `cycle_count` vs `total_energy_throughput_kwh` sebesar `0,984`, dan suhu lingkungan vs suhu puncak *fast charge* sebesar `0,920`. | Menguji resep `compact_log_sensor_caps` (63 neuron input) yang membuang `total_energy_throughput_kwh` dan `peak_temp_during_fast_charge_c`. | Fitur berkorelasi marginal rendah tidak dibuang tanpa uji validasi karena MLP mampu menangkap interaksi nonlinear. |

### 6.2 Implikasi terhadap Pelatihan Backpropagation dan Protokol Evaluasi

Temuan-temuan pada BAB III dan BAB IV juga mengarahkan desain eksperimen pelatihan jaringan saraf tiruan pada [training.py](../../src/ev_battery/training.py) dan protokol evaluasi akhir pada [evaluation.py](../../src/ev_battery/evaluation.py):

1. **Kebutuhan Aktivasi Nonlinear dan Eksplorasi Kapasitas Jaringan:** Perbedaan kemiringan kurva degradasi SOH antarkimia baterai (LFP, NMC, NCA, dan LMO) serta hubungan monoton-melengkung antara siklus, umur, dan SOH menuntut jaringan MLP dengan fungsi aktivasi nonlinear pada lapisan tersembunyi (diuji antara `sigmoid`, `tanh`, dan `ReLU`) serta variasi arsitektur lapisan tersembunyi (`[2]`, `[8]`, `[32]`, `[64]`, `[64, 32]`, hingga `[128, 64, 32]`) yang dipasangkan dengan satu neuron output linear.
2. **Penanganan Ketimpangan Distribusi Target Kontinu:** Karena 51,622% sampel *train* menumpuk pada kondisi sehat (`SOH >= 95%`) dan hanya 1,503% berada pada kondisi kritis (`SOH < 70%`), eksperimen menguji skema pembobotan sampel pada gradien fungsi *loss* berdasarkan kebalikan frekuensi pita SOH (`inverse` dan `inverse_sqrt`) sebagai padanan regresi dari penyeimbangan kelas, sekaligus membandingkan fungsi *loss* MSE, MAE, dan Huber.
3. **Pelaporan Evaluasi Terpilah per Pita SOH dan per Baterai:** Metrik utama dilaporkan kembali ke skala asli poin persentase SOH menggunakan MAE dan RMSE (bukan akurasi klasifikasi). Untuk mencegah tertipu oleh rendahnya galat pada baterai sehat, evaluasi pada himpunan *test* yang telah dikunci wajib menyajikan rincian galat pada kelima pita SOH (`<70%`, `70-<80%`, `80-<90%`, `90-<95%`, dan `>=95%`) serta ringkasan galat per unit `battery_id`. Sebagaimana tercatat pada hasil akhir di [README.md](../../README.md), model *baseline* (`67 -> [32] sigmoid -> 1 linear`, 2.209 parameter) mencapai Test MAE `0,96018 +/- 0,00641` poin persentase dan model kandidat (`67 -> [8] sigmoid -> 1 linear`, 553 parameter) mencapai Test MAE `0,97268 +/- 0,00325` poin persentase, dengan kelemahan terbesar keduanya berada tepat pada pita langka `SOH < 70%` yang telah teridentifikasi sejak tahap EDA.

### 6.3 Keterbatasan Studi dan Kaidah Reproduksibilitas

Laporan audit dan EDA ini tunduk pada sejumlah batasan metodologis yang dijaga secara transparan. Pertama, seluruh pola statistik yang ditemukan merepresentasikan perilaku generator sintetis, sehingga tidak boleh digeneralisasikan sebagai hukum degradasi elektrokimia pada armada kendaraan listrik komersial. Kedua, ketiadaan kode sumber simulator serta ketiadaan *mask* asli untuk *missing values* dan *outliers* membuat pemisahan antara *noise* buatan dan sinyal ekstrem hanya dapat dievaluasi secara tidak langsung melalui uji sensitivitas dan validasi silang. Ketiga, jendela waktu agregasi untuk fitur-fitur rata-rata (seperti `avg_daily_distance_km` dan `avg_charging_power_kw`) tidak didokumentasikan oleh pembuat data, sehingga seluruh fitur diasumsikan telah tersedia pada saat titik observasi berlangsung.

Seluruh tabel dan ke-18 grafik beresolusi 140 dpi di dalam laporan ini dihasilkan secara deterministik tanpa memodifikasi berkas mentah dan terikat secara kriptografis melalui berkas `results/audit/provenance.json` serta `results/eda/eda_provenance.json` (menggunakan lingkungan Python 3.13.1, NumPy 2.2.5, Pandas 2.3.2, Matplotlib 3.10.6, dan SciPy 1.16.1). Seluruh tahapan audit dan eksplorasi dapat direproduksi kapan saja dari direktori akar proyek melalui perintah `make audit && make eda` atau dengan mengeksekusi ulang [00_data_audit.ipynb](../../notebooks/00_data_audit.ipynb) dan [01_eda.ipynb](../../notebooks/01_eda.ipynb).
