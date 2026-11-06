# Panduan Proyek JST: EV Battery Health

Dokumen kerja kelompok untuk EDA, preprocessing, dan eksperimen Multi Layer Perceptron (MLP) yang dilatih dengan backpropagation.

> Konteks kuliah: proyek akhir Jaringan Saraf Tiruan. Model eksperimen dibatasi pada Multi Layer Perceptron (MLP) dengan pelatihan backpropagation. Model JST lain yang telah dipelajari hanya menjadi landasan teori.
>
> Status dokumen: panduan kerja dengan audit data awal yang sudah dijalankan pada 6 November 2026. Belum ada hasil training atau evaluasi model. Persyaratan yang berasal dari tugas, rekomendasi metodologis, dan tambahan bersyarat dibedakan di bagian 29.

## Ringkasan keputusan audit — 6 November 2026

- [Notebook audit](notebooks/00_data_audit.ipynb) sudah dieksekusi, dengan tabel dan empat grafik awal train.
- [Laporan audit](results/audit/AUDIT_REPORT.md) dan [kartu dataset](docs/DATASET_CARD.md) memuat bukti serta batas kesimpulan.
- Target: estimasi SOH saat ini (`state_of_health_pct`) untuk baterai baru, bukan forecasting/RUL.
- Split berdasarkan battery_id: **34.929 baris train / 7.499 validation / 7.572 test**, tanpa baterai tumpang tindih. [Manifest](data/splits/battery_split.csv) memakai seed 42.
- capacity_fade_pct adalah tepat 100 − SOH pada 48.131 nilai terisi; dilarang menjadi input. Tiga target alternatif juga dikeluarkan.
- Ada 21 kandidat fitur mentah baseline: 15 numerik, 5 kategori, 1 biner. Jumlah neuron input baru dihitung setelah encoding. Empat kandidat sensor ditahan untuk ablation terpisah karena risiko proxy sintetis.
- Timesteps ditunda: observation_date hilang pada 2.216 baris, dan counter siklus/energi tidak selalu monoton. Missing tanggal servis sebagian bersifat struktural.
- Belum ada preprocessing fit, training, atau skor model. EDA lengkap setiap fitur adalah tahap berikutnya.

## Cara memakai panduan ini

- **Untuk memahami tugas:** baca bagian 1–3, 15–16, dan 29.
- **Untuk mulai bekerja:** isi lembar keputusan pada bagian 17, lalu ikuti urutan kerja bagian 11.
- **Untuk mengerjakan data:** gunakan bagian 4–6, 17–18.
- **Untuk memahami jaringan dan algoritma:** pelajari bagian 19–20, termasuk contoh hitung manual.
- **Untuk menjalankan eksperimen:** gunakan protokol rinci bagian 21–23 sebagai acuan pelaksanaan bagian 7–8.
- **Untuk membagi pekerjaan dan menulis hasil:** gunakan bagian 24–26.
- **Untuk memeriksa kesiapan pengumpulan:** gunakan bagian 27–30.

Bagian 1–14 memberi gambaran alur lengkap; bagian 15–30 memperinci pelaksanaan dan materi JST. Jika ada pilihan rancangan yang lebih spesifik, ikuti protokol rinci bagian 21–22 dan catat keputusan aktual kelompok.

## 1. Tujuan dan batasan

Tujuan utama: membangun dan mengevaluasi MLP untuk memprediksi kesehatan baterai EV dari informasi operasional yang tersedia, lalu menjelaskan pengaruh arsitektur dan pengaturan pelatihan terhadap kualitas prediksi.

Sumber: https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset
File yang dirujuk: ev_battery_health.csv.

Status terkini: CSV telah diaudit. Terdapat 50.000 observasi, 35 kolom, dan 2.600 baterai. Target kerja adalah state_of_health_pct dalam persen. Dokumentasi menyebut data sintetis berlisensi CC0. Lihat kartu dataset dan laporan audit untuk fakta, batasan, dan daftar fitur aktual; bagian umum berikut tetap menjadi panduan metode.

Batasan wajib:
- Model prediksi hanya MLP/feedforward neural network dengan backpropagation. Jangan memakai CNN, RNN, LSTM, Transformer, Random Forest, SVM, atau boosting.
- Optimizer seperti SGD dan Adam memperbarui bobot menggunakan gradien hasil backpropagation; keduanya dapat dibandingkan bila diperbolehkan dosen.
- Semua hasil harus berasal dari eksekusi nyata dan dapat direproduksi.
- Jangan menyimpulkan hubungan sebab akibat hanya dari korelasi atau prediksi.

## 2. Tentukan masalah sebelum eksperimen

Keputusan kerja setelah audit: regresi untuk memperkirakan state_of_health_pct pada saat observasi, dengan generalisasi ke baterai yang tidak ada dalam train. Tidak mengklaim forecasting atau prediksi sisa umur.

Gunakan keputusan aktual pada bagian 17 dan kontrak fitur sebagai acuan. Tabel berikut menjelaskan pertanyaan yang harus tetap dapat dijawab kelompok:

| Keputusan | Yang harus dicatat |
|---|---|
| Unit observasi | Satu baris mewakili baterai, siklus, sesi, atau pengukuran? |
| Target | Nama kolom, definisi, satuan, rentang, dan cara pengukurannya |
| Saat prediksi | Kapan prediksi dibuat dan informasi apa yang sudah tersedia saat itu? |
| Fitur | Kolom yang boleh diketahui pada saat prediksi |
| Struktur data | Independen, berulang per baterai, atau deret waktu |
| Tujuan generalisasi | Baterai baru atau kondisi masa depan baterai yang sama |

Jangan otomatis membuat kelas sehat/tidak sehat dari target numerik. Jika tugas mengharuskan klasifikasi, tetapkan batas kelas berdasarkan sumber atau ketentuan dosen, jelaskan alasannya, dan gunakan metrik klasifikasi. Regresi dan klasifikasi bukan dua hasil yang bisa dibandingkan langsung.

## 3. Istilah yang dipakai kelompok

- SOH: kondisi kesehatan baterai relatif terhadap kondisi acuannya; ikuti definisi dataset. SOH berbasis kapasitas sering dinyatakan sebagai kapasitas saat ini dibagi kapasitas awal, tetapi jangan menganggap rumus ini pasti digunakan dataset.
- SOC: tingkat keterisian baterai saat pengukuran; berbeda dengan SOH.
- Degradasi: penurunan kemampuan baterai terhadap pemakaian atau waktu.
- Fitur: informasi masukan model; target: nilai yang diprediksi.
- Leakage/kebocoran data: informasi target, masa depan, atau data evaluasi ikut memengaruhi pelatihan.
- Epoch: satu putaran pembelajaran atas seluruh data train; batch: bagian data dalam satu pembaruan bobot.
- Backpropagation: menghitung gradien loss terhadap bobot; optimizer: menggunakan gradien tersebut untuk memperbarui bobot.
- Train: belajar parameter; validation: memilih konfigurasi; test: evaluasi akhir yang dikunci.

## 4. Tahap A — Audit data dan EDA

Checklist pekerjaan:
- Simpan CSV asli tanpa perubahan dan catat tanggal unduh, versi sumber, lisensi jika tersedia, serta asal pengumpulan/sintesis data.
- Tampilkan ukuran data, contoh baris, tipe data, nama kolom, satuan, dan statistik deskriptif.
- Buat kamus data: kolom, arti, tipe, satuan, rentang wajar, peran fitur/target/ID/waktu.
- Hitung missing value per kolom, duplikasi baris, nilai unik, dan nilai tidak valid seperti satuan campuran atau angka tak masuk akal.
- Identifikasi ID baterai dan timestamp/siklus. Periksa urutan waktu, jumlah pengukuran per baterai, dan apakah pengamatan berulang.
- Periksa distribusi target dan fitur numerik, frekuensi kategori, serta outlier menggunakan histogram, boxplot, dan ringkasan kuantil.
- Periksa hubungan fitur-target dengan scatterplot dan korelasi numerik. Korelasi tinggi antarf fitur menjadi bahan evaluasi redundansi.
- Jika ada waktu/siklus, plot perkembangan target per baterai. Jangan menghubungkan baris yang berasal dari baterai berbeda.
- Audit kebocoran: fitur turunan langsung target, pengukuran sesudah saat prediksi, dan ID yang hanya menghafal observasi.

Pemisahan untuk mengidentifikasi kelompok/waktu dilakukan sejak awal. Analisis yang memengaruhi pemilihan fitur dan transformasi dilakukan pada train. Hindari memakai pola test untuk merancang model.

Hasil tahap ini: kamus data, grafik dengan interpretasi, daftar masalah kualitas data, dan keputusan target serta skema pembagian data. Setiap grafik harus menjawab pertanyaan, bukan sekadar memenuhi jumlah gambar.

## 5. Tahap B — Pembagian data dan preprocessing

### Pembagian data

Gunakan sekitar 70% train, 15% validation, 15% test sebagai titik awal, disesuaikan dengan ukuran dan struktur data.
- Baris independen: random split dengan seed tetap.
- Pengukuran berulang dan tujuan baterai baru: split berdasarkan ID baterai agar ID tidak melintasi subset.
- Tujuan prediksi masa depan: split kronologis; semua fitur harus tersedia sebelum waktu target.
- Jika ada kelompok sekaligus waktu, dokumentasikan bagaimana keduanya dijaga sesuai tujuan evaluasi.
- Untuk data kecil, pertimbangkan cross-validation pada bagian pengembangan dengan aturan kelompok/waktu yang sama. Test tetap disimpan untuk akhir.

### Pipeline preprocessing

1. Perbaiki tipe, format, dan satuan berdasarkan aturan terdokumentasi.
2. Hapus duplikasi yang terbukti tidak sah. Nilai sama pada waktu berbeda belum tentu duplikasi.
3. Baris tanpa target tidak digunakan untuk supervised learning; laporkan jumlahnya. Jangan mengimputasi target sebagai label latihan.
4. Imputasi fitur numerik dengan median train; kategori dengan modus train atau kategori missing yang eksplisit. Tambahkan indikator missing jika ada alasan.
5. Tangani outlier berdasarkan pengetahuan domain. Jangan menghapus semua titik ekstrem: kondisi baterai buruk justru mungkin penting. Batas berbasis statistik dipelajari dari train.
6. Encode kategori nominal dengan one-hot encoding dan dukung kategori baru. ID biasanya dipakai untuk split, bukan fitur.
7. StandardScaler sebagai default numerik; MinMaxScaler sebagai pembanding terkontrol. Fit hanya pada train, lalu transform validation dan test.
8. Feature selection: buang kebocoran, kolom konstan, ID tanpa makna prediktif, serta redundansi yang dapat dijustifikasi. Seleksi berbasis target wajib dilakukan dalam train/fold train.
9. Simpan daftar fitur akhir dan urutan kolom. Jumlah neuron input dihitung setelah encoding dan seleksi.

### Apakah perlu balancing?

- Regresi: tidak memakai SMOTE kelas secara otomatis. Periksa apakah target jarang di rentang tertentu dan laporkan error per rentang SOH. Pembobotan sampel boleh menjadi eksperimen tambahan yang dijustifikasi.
- Klasifikasi: cek proporsi kelas pada train. Bandingkan tanpa balancing dan class weighting. Jika memakai resampling, lakukan hanya pada train, sesudah split, dengan metode yang cocok untuk tipe fiturnya.
- Jangan menyeimbangkan validation/test; pertahankan distribusi evaluasi yang sesuai penggunaan.

Hasil tahap ini: pipeline yang bisa dipakai ulang, alasan setiap transformasi, ukuran subset, dan bukti tidak ada kebocoran.

## 6. Input fitur atau timesteps

Default: MLP tabular, satu sampel berupa vektor p fitur hasil preprocessing; neuron input = p.

Timesteps hanya dipakai jika tersedia urutan pengukuran nyata beserta ID baterai dan waktu/siklus. Untuk jendela T langkah dengan p fitur per langkah, flatten menjadi T × p neuron input. Contoh T=5 dan p=8 menghasilkan 40 neuron input. Ini tetap MLP, bukan model recurrent.

Tetapkan apakah target adalah kesehatan pada akhir jendela atau h langkah ke depan. Bentuk jendela dalam masing-masing subset dan masing-masing baterai; jangan melintasi ID atau batas split. Jangan membuat urutan dari urutan baris CSV tanpa bukti waktu. Jika data tidak berurutan, timesteps tidak berlaku dan cukup dijelaskan alasannya.

## 7. Tahap C — Rancangan eksperimen

### Baseline MLP yang disarankan

- Input: seluruh fitur valid setelah preprocessing.
- Hidden layers: 64 lalu 32 neuron.
- Aktivasi hidden: ReLU.
- Regresi: satu neuron output linear, loss MSE; metrik MAE, RMSE, dan R².
- Klasifikasi biner bila dipilih: satu output sigmoid dengan binary cross-entropy, atau output logits dengan loss yang sesuai library. Jangan menerapkan sigmoid dua kali.
- Multikelas bila dipilih: output sebanyak kelas dengan cross-entropy dan format target yang sesuai library.
- Optimizer: Adam, learning rate awal 0,001, batch size 32.
- Epoch maksimum: 200; early stopping berdasarkan validation MAE dalam skala target asli dengan patience 15 dan pemulihan bobot terbaik. Lihat bagian 21.3 untuk protokol perbandingan loss dan epoch.
- Seed awal tetap; konfigurasi final diulang sedikitnya tiga seed.

Angka di atas merupakan titik awal yang perlu diuji, bukan konfigurasi yang sudah terbukti terbaik.

### Matriks eksperimen bertahap

| Tahap | Variasi | Tujuan |
|---|---|---|
| E0 | Baseline di atas | Acuan awal |
| E1 | Fitur valid lengkap vs fitur terseleksi | Pengaruh pilihan/jumlah input |
| E2 | [32], [64,32], [128,64,32] | Pengaruh lebar dan kedalaman |
| E3 | ReLU vs tanh vs sigmoid pada hidden layer | Pengaruh aktivasi |
| E4 regresi | MSE vs MAE vs Huber | Pengaruh loss |
| E5 | Maksimum epoch 50, 100, 200 | Pengaruh durasi belajar |
| E6 opsional | StandardScaler vs MinMaxScaler; learning rate; regularisasi | Perbaikan terarah |
| E7 bersyarat | T=1, 3, 5, 10 | Pengaruh panjang riwayat jika data berurutan |

Ubah satu faktor per tahap, pertahankan faktor lain, dan lanjutkan dari konfigurasi terbaik berdasarkan validation. Ini pencarian bertahap, bukan bukti bahwa semua interaksi kombinasi telah diuji. Untuk klasifikasi, gunakan loss yang sesuai label; jangan mengganti cross-entropy dengan loss regresi hanya demi variasi.

Pada E5, gunakan kebijakan early stopping yang sama dan catat epoch aktual; jika semua berhenti lebih awal pada epoch yang sama, eksperimen tidak menunjukkan perbedaan jumlah pembelajaran yang nyata. Jika ingin mengisolasi anggaran epoch, jalankan perbandingan terpisah dengan early stopping nonaktif dan simpan hasilnya secara jelas.

Jangan membandingkan angka validation loss lintas jenis loss untuk memilih model. Gunakan satu metrik validation yang sama, misalnya MAE untuk regresi. Saat mempelajari fitur/timesteps, hitung ulang dimensi input dan jumlah sampel yang tersedia.

## 8. Evaluasi dan analisis

Regresi:
- Tetapkan MAE sebagai metrik utama, RMSE dan R² sebagai pelengkap.
- Sajikan metrik dalam satuan target asli. Jika target diskalakan, kembalikan skala sebelum evaluasi.
- Plot train/validation loss, aktual vs prediksi, serta distribusi residual.
- Analisis error pada rentang kesehatan baterai dan kelompok kondisi yang relevan, dengan jumlah sampel tiap kelompok.

Klasifikasi bila dipilih:
- Gunakan macro-F1 atau metrik yang sesuai ketimpangan kelas, confusion matrix, serta precision/recall per kelas; accuracy menjadi pelengkap.
- Pilih ambang prediksi pada validation dan kunci sebelum test.

Untuk semua skenario:
- Bandingkan dengan baseline konstan sederhana (rata-rata/median train untuk regresi atau kelas mayoritas untuk klasifikasi) sebagai pemeriksaan kewajaran, bukan model pesaing di luar batas tugas.
- Pilih konfigurasi dengan validation, kunci pipeline dan konfigurasi, kemudian evaluasi test pada akhir. Jangan memilih seed terbaik berdasarkan test.
- Laporkan rata-rata dan standar deviasi hasil sedikitnya tiga seed final. Jika memakai ensemble, nyatakan sebagai eksperimen terpisah.
- Jelaskan underfitting/overfitting dari kurva dan selisih train-validation, bukan dugaan.
- Bahas keterbatasan sumber data, representasi kondisi nyata, ukuran sampel, dan generalisasi.

## 9. Catatan eksperimen wajib

Satu baris per run berisi:
ID run, versi data, split/seed, target, fitur, jumlah input, timesteps bila ada, ukuran train/validation/test, arsitektur, aktivasi hidden/output, loss, optimizer, learning rate, batch size, epoch maksimum, epoch aktual/terbaik, regularisasi, metrik train/validation, waktu pelatihan, dan lokasi model.

Metrik test hanya diisi untuk evaluasi final. Simpan kurva, konfigurasi, preprocessing, bobot model, serta versi library. Hindari menyimpan hasil hanya dalam screenshot.

## 10. Pembagian kerja kelompok

| Peran | Tanggung jawab | Hasil untuk diserahkan |
|---|---|---|
| Data dan EDA | Audit sumber, kamus data, kualitas, visualisasi | Notebook EDA dan temuan |
| Preprocessing | Split, leakage audit, transformasi, seleksi fitur | Pipeline dan daftar fitur |
| Modeling | Baseline, matriks eksperimen, pencatatan hasil | Notebook/script eksperimen dan model |
| Evaluasi dan laporan | Analisis error, tabel perbandingan, pembahasan | Laporan, slide, kesimpulan |

Peran boleh dirangkap sesuai jumlah anggota. Semua anggota harus memahami target, alasan split, alur backpropagation, dan alasan pemilihan model. Sepakati satu split bersama sebelum menjalankan eksperimen agar hasil dapat dibandingkan.

## 11. Urutan kerja dan titik pemeriksaan

1. Unduh dan audit data → setujui kamus data, target, dan unit observasi.
2. Tetapkan split dan kunci test → setujui daftar risiko leakage.
3. Selesaikan EDA train dan preprocessing → verifikasi dimensi serta nilai keluaran pipeline.
4. Jalankan baseline MLP sampai menghasilkan metrik dan kurva yang benar.
5. Jalankan eksperimen bertahap dan catat seluruh run, termasuk hasil buruk.
6. Pilih konfigurasi dengan validation dan ulangi seed final.
7. Jalankan evaluasi test akhir, analisis error, dan tulis keterbatasan.
8. Rapikan notebook, laporan, slide, serta petunjuk menjalankan ulang.

## 12. Struktur hasil yang disarankan

- data/raw/: data asli jika lisensi mengizinkan distribusi; selain itu petunjuk unduh.
- notebooks/: audit/EDA, preprocessing, eksperimen, dan evaluasi.
- src/: fungsi pipeline dan pelatihan yang dipakai ulang jika diperlukan.
- results/: tabel eksperimen, konfigurasi, dan grafik.
- models/: model final dan preprocessing yang cocok.
- report/: laporan dan presentasi.
- README.md: petunjuk instalasi dan urutan menjalankan.
- CONTEXT.md: panduan kerja ini, diperbarui setelah keputusan dikonfirmasi.

## 13. Kriteria tugas selesai

- [ ] Target dan pertanyaan penelitian jelas, sesuai kolom nyata.
- [ ] Data dan satuan terdokumentasi; semua grafik utama memiliki interpretasi.
- [ ] Missing value, outlier, duplikasi, normalisasi, feature selection, dan balancing dibahas, termasuk alasan jika tidak dilakukan.
- [ ] Split sesuai struktur data dan semua transformasi belajar hanya dari train.
- [ ] Model hanya MLP dengan backpropagation; dimensi input/output dapat dijelaskan.
- [ ] Eksperimen mencakup input, hidden layer, aktivasi, loss yang sesuai, dan epoch.
- [ ] Semua konfigurasi dibandingkan pada split serta metrik yang konsisten.
- [ ] Model dipilih menggunakan validation dan test digunakan pada evaluasi akhir.
- [ ] Hasil final mencakup variasi seed, analisis error, dan keterbatasan.
- [ ] Model, pipeline, konfigurasi, tabel hasil, laporan, dan petunjuk reproduksi tersedia.
- [ ] Semua angka laporan sesuai hasil yang tersimpan.

## 14. Pertanyaan untuk diselesaikan pada rapat awal

1. Apakah dosen menetapkan target tertentu atau mengizinkan regresi SOH?
2. Apa definisi dan satuan target dalam CSV sebenarnya?
3. Apakah setiap baterai memiliki beberapa pengamatan berurutan?
4. Apakah tujuan evaluasi baterai baru atau masa depan baterai yang sama?
5. Apakah ada ketentuan framework, jumlah eksperimen, format laporan, dan tenggat?
6. Siapa pemilik tiap hasil dan kapan titik pemeriksaan internalnya?

Dokumen ini berisi rancangan pelaksanaan dan keputusan audit awal; hasil eksperimen model belum tersedia. Perbarui keputusan bila EDA train/preprocessing mengungkap kebutuhan perubahan, dan jangan menggunakan test untuk menentukannya.

---

# Pendalaman untuk Proyek Akhir Jaringan Saraf Tiruan

Bagian berikut melengkapi bagian 1–14. Fokusnya adalah apa yang harus dipahami, dikerjakan, diuji, dan dipertanggungjawabkan saat presentasi. Angka konfigurasi adalah rancangan awal; hasil eksperimen belum tersedia.

## 15. Posisi proyek terhadap materi kuliah

| Materi yang sudah dipelajari | Hubungannya dengan proyek | Batas penggunaannya |
|---|---|---|
| McCulloch–Pitts | Dasar neuron menerima input, menggabungkan input, lalu menghasilkan output | Landasan teori; bukan model eksperimen |
| Hebb Net | Pengenalan perubahan bobot melalui pembelajaran | Bandingkan konsep aturannya dengan gradien; jangan digunakan untuk melatih model proyek |
| Decision boundary | Memahami bagaimana model memisahkan kelas | Relevan jika klasifikasi; untuk regresi bahas fungsi/permukaan prediksi |
| Perceptron | Bobot, bias, weighted sum, dan batas model linear | Landasan menuju MLP; bukan algoritma pelatihan utama proyek |
| Adaline/Madaline | Error numerik dan hubungan pembelajaran dengan optimasi | Landasan perbandingan konseptual; tidak menjadi model pembanding |
| Self-Organizing Maps | Pembelajaran tanpa label dan pemetaan kemiripan | Tidak dipakai sebagai model maupun tahap clustering dalam lingkup utama |
| Learning Vector Quantization | Pembelajaran berbasis prototipe | Tidak dipakai dalam eksperimen |
| Backpropagation | Perhitungan gradien pada jaringan berlapis dengan chain rule | Wajib menjadi dasar pelatihan seluruh MLP |
| Extreme Learning Machine | Perbedaan strategi pelatihan hidden/output dengan MLP backpropagation | Boleh dijelaskan secara konseptual; tidak dijalankan sebagai model proyek |

MLP adalah arsitektur jaringan. Backpropagation adalah cara menghitung gradien. SGD/Adam adalah aturan memperbarui parameter dari gradien tersebut. Ketiga istilah ini tidak boleh dipakai seolah-olah sinonim.

MLP pada proyek ini menggunakan koneksi feedforward: input → hidden layer satu atau lebih → output. Model tidak memelihara state recurrent. Hidden layer memakai aktivasi nonlinear. Tanpa nonlinearitas di antara layer, komposisi beberapa transformasi linear/affine tetap merupakan transformasi affine dan tidak memberi kemampuan nonlinear yang diharapkan.

Target pembelajaran kelompok:
1. Mampu menggambar jaringan dengan dimensi yang benar.
2. Mampu menjelaskan satu forward pass dan satu pembaruan bobot.
3. Mampu membedakan loss untuk pelatihan dan metrik untuk evaluasi.
4. Mampu menjelaskan mengapa pemisahan data dan normalisasi memengaruhi kesahihan hasil.
5. Mampu menyimpulkan eksperimen dari bukti, termasuk saat perubahan konfigurasi tidak memperbaiki hasil.

## 16. Rumusan masalah dan pertanyaan penelitian

Judul kerja yang aman sebelum audit data:
“Analisis Pengaruh Konfigurasi Multi Layer Perceptron dengan Backpropagation untuk Prediksi Kesehatan Baterai Kendaraan Listrik”.

Setelah target diverifikasi, ganti “kesehatan” dengan nama target yang benar. Jika targetnya SOH saat ini, jangan menulis bahwa model meramal sisa umur atau kondisi masa depan. Prediksi SOH, prediksi degradasi ke depan, dan Remaining Useful Life merupakan masalah berbeda.

Rumusan masalah utama:
- Seberapa baik MLP memprediksi target pada data yang belum digunakan untuk pelatihan?
- Bagaimana pemilihan fitur, jumlah hidden layer/neuron, aktivasi, loss, dan jumlah epoch memengaruhi hasil?
- Pada kondisi apa prediksi model paling sering meleset?

Pertanyaan tambahan bersyarat:
- Apakah riwayat beberapa pengukuran meningkatkan prediksi jika tersedia data berurutan yang sah?
- Apakah regularisasi atau perubahan learning rate membantu ketika terlihat overfitting atau pembelajaran tidak stabil?

Hipotesis adalah dugaan yang diuji, bukan kesimpulan awal. Contoh: “Normalisasi diperkirakan membantu optimasi karena rentang fitur berbeda.” Hasil boleh mendukung atau tidak mendukung dugaan tersebut.

Tabel tujuan–bukti yang harus disiapkan:

| Tujuan | Bukti yang harus tersedia |
|---|---|
| Memahami data | Kamus data, audit kualitas, grafik dan interpretasi |
| Membuat input yang sah | Pipeline, daftar fitur, catatan split, audit leakage |
| Membangun MLP backpropagation | Diagram, konfigurasi, penjelasan forward/backward |
| Mengukur pengaruh konfigurasi | Tabel eksperimen terkontrol dan grafik perbandingan |
| Menilai generalisasi | Evaluasi test final dan analisis error |
| Memungkinkan reproduksi | Instruksi eksekusi, versi lingkungan, seed, artefak |

## 17. Audit dataset yang lebih rinci

### 17.1 Lembar keputusan data

Isi bersama sebelum coding model:

| Isian | Nilai aktual |
|---|---|
| Tanggal unduh dan versi dataset | 6 November 2026; SHA-256 di results/audit/provenance.json |
| Jumlah baris/kolom asli | 50.000 baris / 35 kolom; 2.600 baterai |
| Sumber nyata/simulasi/sintetis/tidak dijelaskan | Sepenuhnya sintetis menurut README sumber; lisensi CC0 |
| Satu baris mewakili | Satu observasi baterai; observation_id unik |
| Nama dan satuan target | state_of_health_pct; persen; kapasitas saat ini relatif nominal menurut kamus |
| Arti target: saat ini/masa depan | SOH saat observasi, bukan masa depan |
| Kolom ID baterai | battery_id |
| Kolom waktu/siklus dan intervalnya | observation_date, battery_age_days, cycle_count; interval tidak seragam; 2.216 tanggal kosong; counter perlu audit lanjutan |
| Daftar kolom yang tersedia saat prediksi | 21 kandidat di configs/feature_contract.json; availability ringkasan saat observasi adalah asumsi yang dicatat |
| Tujuan generalisasi | Baterai yang belum terlihat dalam train, pada skenario estimasi SOH saat ini |
| Aturan split dan seed | Group berdasarkan battery_id; NumPy default_rng(42); 1.820/390/390 baterai train/validation/test; manifest dikunci |
| Metrik utama dan arah optimasi | Rekomendasi regresi: MAE, lebih kecil lebih baik |

Jika deskripsi sumber tidak menjelaskan asal data, tulis “asal pengumpulan tidak diketahui dari dokumentasi yang tersedia”. Pola data yang rapi saja tidak cukup untuk menyatakan bahwa data sintetis.

### 17.2 Template kamus data

Satu baris per kolom dengan isian:
- Nama persis dalam CSV.
- Nama yang mudah dipahami.
- Definisi dan sumber definisinya.
- Tipe asli dan tipe setelah perbaikan.
- Satuan dan rentang yang teramati.
- Jumlah missing dan nilai unik.
- Peran: target, fitur numerik, fitur kategori, ID, waktu, atau dikeluarkan.
- Apakah sudah tersedia saat prediksi.
- Transformasi yang dipilih dan alasannya.

Jangan mengisi satuan dari tebakan nama kolom. Bedakan satuan persen 0–100 dengan rasio 0–1 dan periksa konsistensinya.

### 17.3 Pemeriksaan kebocoran dengan skenario nyata

Contoh hipotetis: jika target SOH dihitung sebagai kapasitas saat ini / kapasitas awal, memasukkan kedua kapasitas itu dapat membuat tugas sekadar merekonstruksi rumus. Nilai error yang sangat kecil belum tentu menunjukkan prediksi kesehatan yang berguna. Tentukan apakah tujuan tugas memang menghitung ulang SOH atau memperkirakannya dari informasi lain, lalu dokumentasikan keputusan fitur.

Contoh lain: ingin memprediksi SOH sebelum sesi pengisian, tetapi memasukkan ringkasan yang baru diketahui setelah sesi selesai. Kolom tersebut tidak sah untuk skenario itu.

ID baterai yang sama dapat menghasilkan banyak baris sangat mirip. Random split baris bisa terlihat bagus tetapi tidak menguji kemampuan pada baterai baru. Keputusan split harus mengikuti pertanyaan penelitian, bukan memilih split yang menghasilkan skor tertinggi.

### 17.4 Daftar grafik EDA dan pertanyaannya

| Grafik/tabel | Pertanyaan yang dijawab | Tindak lanjut |
|---|---|---|
| Statistik dan persentase missing | Kolom mana bermasalah? | Pilih perbaikan/imputasi atau alasan pengeluaran |
| Histogram target | Apakah nilai rendah/tinggi cukup terwakili? | Tentukan analisis error per rentang |
| Histogram/boxplot fitur | Apakah skala berbeda atau ada nilai ekstrem? | Normalisasi dan verifikasi nilai |
| Bar chart kategori | Adakah kategori langka atau salah eja? | Aturan pembersihan kategori |
| Scatter fitur-target | Apakah terlihat pola nonlinear atau kelompok? | Hipotesis yang diuji, bukan kausalitas |
| Korelasi numerik | Adakah redundansi dan potensi turunan target? | Audit fitur lebih lanjut |
| Jumlah pengamatan per ID | Apakah beberapa baterai mendominasi? | Pertimbangkan evaluasi per baterai |
| Target terhadap waktu/siklus | Adakah urutan yang relevan? | Validasi desain temporal |

Grafik berbasis ID/waktu hanya dibuat jika kolomnya benar-benar ada. Sertakan judul, label sumbu, satuan, dan jumlah sampel yang relevan.

## 18. Preprocessing: keputusan, bukan ritual

Tidak semua langkah harus diterapkan. Yang wajib adalah memeriksa kebutuhan, mengambil keputusan, dan memberi alasan.

### 18.1 Missing value

Bedakan nilai kosong sebenarnya, simbol seperti “?”, dan nilai 0 yang sah. Jangan mengganti semua 0 menjadi missing. Laporkan jumlah baris sebelum dan sesudah pengeluaran data. Jika satu kolom sangat banyak kosong, pertimbangkan manfaatnya, ketersediaannya saat penggunaan, dan alternatif pengeluaran; jangan memakai persentase ambang tanpa alasan.

Untuk data temporal, pengisian menggunakan informasi masa depan dapat bocor. Interpolasi atau backward fill tidak otomatis aman. Jika memakai forward fill, batasi dalam satu baterai dan pastikan hanya informasi lampau yang dipakai.

### 18.2 Normalisasi

Standardisasi:

    x_scaled = (x - mean_train) / std_train

Min–max:

    x_scaled = (x - min_train) / (max_train - min_train)

Parameter dihitung per fitur dari train. Fitur konstan perlu ditangani agar tidak membagi nol. Nilai validation/test setelah min–max dapat berada di luar 0–1 jika melewati rentang train; ini bukan otomatis kesalahan dan tidak harus dipotong.

Default: skalakan fitur numerik, biarkan indikator one-hot sebagai 0/1. Penskalaan target bersifat opsional dan harus dicatat. Pada regresi multikeluaran, skala target perlu perhatian khusus, tetapi lingkup awal proyek ini satu target.

### 18.3 Feature selection

Urutan yang disarankan:
1. Hilangkan fitur yang tidak sah pada waktu prediksi.
2. Hilangkan fitur yang membocorkan atau merekonstruksi target di luar tujuan tugas.
3. Keluarkan ID administratif dan kolom konstan.
4. Evaluasi redundansi berdasarkan train dan pengetahuan domain.
5. Bandingkan set fitur valid penuh dengan subset yang ditetapkan menggunakan train/validation.

Korelasi linear rendah tidak membuktikan fitur tidak berguna untuk MLP nonlinear. Korelasi tinggi juga tidak otomatis membenarkan penghapusan. Simpan tabel “dipakai/dikeluarkan/alasan” agar keputusan dapat diperiksa.

Jika memakai feature ranking berbasis target, pemilihannya harus diulang dalam setiap fold train saat cross-validation. Jangan menjalankan seleksi sekali pada semua data sebelum cross-validation.

### 18.4 Balancing dan distribusi evaluasi

Untuk regresi, tulis secara eksplisit bahwa balancing kelas tidak diterapkan karena target kontinu. Itu keputusan metodologis yang sah. Periksa error pada target yang jarang terwakili dan jangan menyamarkannya dengan metrik agregat.

Untuk klasifikasi, class weighting lebih sederhana sebagai pembanding awal daripada menambah data sintetis. Bandingkan dengan konfigurasi tanpa weighting menggunakan metrik yang sama. Jangan mengubah test agar proporsi kelas menjadi seimbang.

### 18.5 Pemeriksaan sebelum model dilatih

- Tidak ada NaN atau nilai tak hingga pada input/output model.
- Jumlah baris X dan y sama dan pasangan indeksnya benar.
- Daftar serta urutan fitur train, validation, test sama.
- Tidak ada ID yang melintasi split jika memakai group split.
- Tidak ada jendela yang melintasi batas waktu atau baterai.
- Target tidak muncul di input dan tidak terselip sebagai salinan nama lain.
- Statistik scaler/imputer hanya berasal dari train.
- Jumlah input jaringan sama dengan dimensi keluaran preprocessing.

## 19. Anatomi MLP yang harus dapat dijelaskan

Misalkan p adalah jumlah fitur setelah preprocessing dan h_l jumlah neuron hidden layer ke-l. Untuk satu sampel, gunakan vektor kolom:

    a^(0) = x
    z^(l) = W^(l) a^(l-1) + b^(l)
    a^(l) = f_l(z^(l))
    y_hat = a^(L)

Dimensi:
- W^(l): n_l × n_(l-1).
- b^(l), z^(l), a^(l): n_l × 1.
- L: indeks layer output; input tidak dihitung sebagai layer berbobot.

Contoh hipotetis p=12, hidden [64,32], satu output:

    Input (12) → Dense (64, ReLU) → Dense (32, ReLU) → Output (1, linear)

Jumlah parameter trainable:

    (12×64 + 64) + (64×32 + 32) + (32×1 + 1)
    = 832 + 2080 + 33
    = 2945 parameter

Jumlah ini harus dihitung ulang jika encoding, jumlah fitur, timesteps, atau arsitektur berubah. Banyaknya neuron input bukan jumlah baris data. Banyaknya neuron output bukan jumlah fitur.

Bias memberi pergeseran pada transformasi neuron. Bobot dan bias dipelajari, sedangkan jumlah layer, neuron, aktivasi, learning rate, dan batch size merupakan hyperparameter yang dipilih sebelum atau melalui eksperimen.

### Aktivasi yang dibandingkan

| Aktivasi | Rumus | Karakter yang perlu dibahas |
|---|---|---|
| ReLU | max(0,z) | Gradien 1 untuk z>0 dan 0 untuk z<0; neuron dapat tidak aktif |
| Sigmoid | 1/(1+exp(-z)) | Output 0–1; gradien mengecil saat jenuh |
| Tanh | tanh(z) | Output -1–1; juga dapat mengalami kejenuhan |
| Linear | z | Cocok untuk output regresi tanpa pembatasan eksplisit |

ReLU tidak terdiferensialkan tepat di nol; implementasi memilih nilai gradien/subgradien, umumnya nol. Jangan menyatakan semua aktivasi selalu lebih baik atau lebih buruk tanpa hasil.

Output linear dapat menghasilkan prediksi di luar rentang fisik target. Laporkan jika terjadi. Jangan memotong prediksi setelah melihat test untuk mempercantik hasil; bila pembatasan menjadi bagian rancangan, tetapkan dan evaluasi sejak validation.

## 20. Forward pass, loss, backpropagation, dan update bobot

### 20.1 Alur satu batch

1. Ambil satu batch train.
2. Hitung output setiap layer dengan forward pass.
3. Hitung loss dari prediksi dan target batch.
4. Kosongkan akumulasi gradien dari langkah sebelumnya bila framework memerlukannya.
5. Hitung gradien loss terhadap semua bobot/bias dengan backpropagation.
6. Optimizer memperbarui bobot/bias.
7. Ulangi untuk semua batch; satu putaran lengkap adalah satu epoch.
8. Hitung metrik validation tanpa memperbarui parameter.

Shuffle hanya mengubah urutan batch pada train jika sesuai desain; tidak boleh merusak urutan di dalam satu jendela temporal. Forward pass validation/test tidak melakukan pembelajaran.

### 20.2 Loss regresi dan metrik

Untuk N sampel dan error e_i = y_hat_i - y_i:

    MSE  = (1/N) Σ e_i²
    MAE  = (1/N) Σ |e_i|
    RMSE = sqrt(MSE)
    R²   = 1 - Σ(y_i-y_hat_i)² / Σ(y_i-mean(y))²

MSE memberi penalti lebih kuat pada error besar. MAE memberi penalti linear dan gradiennya tidak mulus pada nol. Huber menggabungkan daerah kuadrat dan linear menggunakan parameter delta yang harus dicatat. Nilai R² bisa negatif; untuk target konstan definisi rasio di atas bermasalah sehingga perilaku library perlu dijelaskan jika kasus itu terjadi.

Jika target SOH bersatuan persen, MAE 2 berarti rata-rata selisih absolut 2 poin persentase, bukan otomatis error relatif 2%. MSE memiliki satuan kuadrat target; MAE dan RMSE memiliki satuan target.

### 20.3 Chain rule pada MLP

Untuk satu sampel dan loss setengah kuadrat:

    ℓ = 0.5 × (y_hat - y)²
    δ^(L) = (y_hat - y) ⊙ f'_L(z^(L))
    δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ f'_l(z^(l))
    ∂ℓ/∂W^(l) = δ^(l) (a^(l-1))^T
    ∂ℓ/∂b^(l) = δ^(l)

Simbol ⊙ berarti perkalian elemen per elemen. Untuk output linear, f'_L=1. Untuk MSE tanpa faktor 0.5, turunan membawa faktor 2; reduksi rata-rata batch juga membawa faktor normalisasi. Konsistenlah antara definisi loss dan turunan yang ditulis.

Update dengan SGD sederhana:

    W ← W - learning_rate × grad_W
    b ← b - learning_rate × grad_b

Adam memakai estimasi momen gradien untuk menentukan pembaruan. Walaupun aturan update berbeda, gradien MLP tetap dihitung melalui backpropagation. Untuk presentasi teori, SGD sederhana paling mudah ditelusuri.

### 20.4 Contoh hitung manual satu langkah

Contoh pembelajaran ini bukan data baterai dan bukan arsitektur final. Gunakan satu input, satu hidden ReLU, dan satu output linear:

    x=2; y=1
    w1=0.5; b1=0; w2=0.4; b2=0
    learning_rate=0.1

Forward:

    z1 = 0.5×2 + 0 = 1
    a1 = ReLU(1) = 1
    y_hat = 0.4×1 + 0 = 0.4
    ℓ = 0.5×(0.4-1)² = 0.18

Backward menggunakan seluruh bobot sebelum update:

    δ_output = 0.4-1 = -0.6
    grad_w2 = -0.6×1 = -0.6
    grad_b2 = -0.6
    δ_hidden = 0.4×(-0.6)×1 = -0.24
    grad_w1 = -0.24×2 = -0.48
    grad_b1 = -0.24

Update serentak secara konseptual:

    w1_baru = 0.548; b1_baru = 0.024
    w2_baru = 0.46;  b2_baru = 0.06

Forward setelah update menghasilkan a1=1.12 dan y_hat=0.5752. Loss menjadi sekitar 0.09023. Contoh ini menunjukkan satu update menurunkan loss pada sampel tersebut; bukan jaminan setiap batch atau validation loss selalu turun.

Kesalahan yang harus dihindari: menghitung δ_hidden menggunakan w2 yang sudah diperbarui. Semua gradien satu langkah harus konsisten terhadap parameter forward pass yang sama.

### 20.5 Implementasi dari nol atau framework

Belum ada ketentuan bahwa backpropagation harus dibuat dari nol. Konfirmasikan ketentuan dosen saat rapat/tanya tugas.
- Jika framework diizinkan: gunakan implementasi MLP yang jelas, tampilkan konfigurasi, dan tetap pahami forward/backward.
- Jika implementasi manual diwajibkan: buat forward, loss, backward, update, batching, dan evaluasi secara eksplisit. Mulai dari contoh kecil sebelum data penuh.
- Untuk implementasi manual, gradient check beda hingga pada jaringan kecil dan parameter yang jauh dari titik tak mulus dapat memeriksa turunan. Ini pemeriksaan implementasi, bukan eksperimen kinerja model.

Jangan memilih cara implementasi yang membuat kelompok tidak bisa menjelaskan algoritma yang dijalankan.

## 21. Paket eksperimen konkret yang dapat dibagi ke anggota

Gunakan paket ini untuk regresi satu target setelah audit CSV selesai. Bila akhirnya klasifikasi, revisi output, loss, metrik, dan baseline terlebih dahulu. Jangan menjalankan paket regresi apa adanya untuk label kelas.

### 21.1 Aturan pembanding yang adil

- Gunakan manifest split yang sama: daftar indeks/ID/waktu anggota setiap subset.
- Gunakan target, horizon, dan metrik utama yang sama.
- Jangan mengubah preprocessing diam-diam antara run.
- Gunakan seed penyaringan yang sama. Ini mengurangi satu sumber variasi, tetapi arsitektur berbeda tetap memiliki inisialisasi yang berbeda bentuknya.
- Catat jumlah parameter dan waktu pelatihan karena konfigurasi lebih besar memerlukan sumber daya berbeda.
- Jika membandingkan timesteps, utamakan timestamp target evaluasi yang sama pada semua T agar perbedaan tidak hanya berasal dari perubahan sampel.
- Perubahan loss tidak dibandingkan berdasarkan angka loss masing-masing. Pemilihan lintas loss memakai validation MAE dalam skala target asli.

### 21.2 Paket utama sekitar 15 konfigurasi unik

| ID | Faktor | Nilai yang dijalankan | Yang dipertahankan |
|---|---|---|---|
| B0 | Baseline | Fitur valid lengkap, [64,32], ReLU, MSE, maksimum 200 epoch | Split, scaler, optimizer, learning rate, batch |
| F1 | Input | Subset fitur terpilih | Konfigurasi B0 selain input |
| H1 | Hidden | [32] | Set input terbaik tahap F |
| H2 | Hidden | [64] | Set input terbaik tahap F |
| H3 | Hidden | [64,32] | Set input terbaik tahap F |
| H4 | Hidden | [128,64,32] | Set input terbaik tahap F |
| A1 | Aktivasi hidden | ReLU | Input dan hidden terbaik sebelumnya |
| A2 | Aktivasi hidden | tanh | Sama dengan A1 selain aktivasi |
| A3 | Aktivasi hidden | sigmoid | Sama dengan A1 selain aktivasi |
| L1 | Loss | MSE | Konfigurasi terbaik sebelumnya |
| L2 | Loss | MAE | Sama dengan L1 selain loss |
| L3 | Loss | Huber, delta dicatat | Sama dengan L1 selain loss |
| E1 | Anggaran epoch | 50 | Konfigurasi terbaik sebelumnya |
| E2 | Anggaran epoch | 100 | Sama dengan E1 selain anggaran |
| E3 | Anggaran epoch | 200 | Sama dengan E1 selain anggaran |

Baris tabel tidak selalu berarti run baru: reuse hasil jika seluruh konfigurasi, data, seed, dan kebijakan training identik. Karena itu jumlah konfigurasi unik dapat kurang dari 15. B0 tidak boleh dihapus saat konfigurasi baru terpilih.

Setiap tahap memilih berdasarkan validation MAE. Jika perbedaan sangat kecil, periksa kestabilan seed dan pertimbangkan model yang lebih sederhana; jangan menyatakan unggul bermakna hanya karena selisih angka desimal.

Untuk pembandingan akhir yang lebih kuat, ulangi baseline B0 dan kandidat final pada seed yang sama, misalnya 42, 123, 2026, dengan split tetap. Seed tersebut mengubah inisialisasi dan pengacakan training, bukan komposisi test. Jika kandidat sama dengan baseline, tidak perlu menggandakan run yang identik.

Seleksi bertahap praktis untuk tugas kuliah, tetapi tidak menguji seluruh interaksi faktor. Nyatakan keterbatasan ini di laporan. Jangan menganggap konfigurasi terbaik yang ditemukan sebagai optimum global.

### 21.3 Early stopping dan pemilihan checkpoint

Agar konsisten untuk perbandingan loss:
- Rekomendasi pemantauan checkpoint dan early stopping: validation MAE dalam skala asli untuk seluruh loss regresi.
- Tetapkan patience, min_delta, dan arah optimasi sebelum run. Contoh patience=15; min_delta harus mempertimbangkan satuan target, jangan dipilih dari test.
- Simpan checkpoint dengan validation MAE terbaik dan catat epoch-nya.
- Tampilkan juga train/validation objective loss agar perilaku optimasi setiap run tetap terlihat.

Pilihan ini memperinci baseline awal yang menyebut validation loss. Validation loss tetap berguna untuk kurva, tetapi metrik bersama lebih mudah menjaga konsistensi ketika MSE, MAE, dan Huber dibandingkan.

Untuk eksperimen epoch, pilih dan tulis salah satu desain:
1. **Anggaran maksimum dengan early stopping:** menilai manfaat menaikkan batas epoch dalam prosedur training yang sama; epoch aktual mungkin sama.
2. **Epoch tetap:** nonaktifkan early stopping dan bandingkan bobot pada akhir epoch 50, 100, dan 200, menggunakan seed serta pengaturan sama. Ini lebih langsung menguji lama training. Checkpoint terbaik boleh dicatat sebagai analisis tambahan, tetapi jangan menggantikannya diam-diam sebagai hasil akhir epoch tetap.

Untuk kebutuhan eksplisit “pengaruh epoch”, desain kedua lebih mudah dijelaskan. Untuk model final, early stopping boleh digunakan kembali dengan kebijakan yang telah ditetapkan.

### 21.4 Eksperimen tambahan hanya jika ada alasan

| Temuan | Tambahan yang relevan |
|---|---|
| Train membaik, validation memburuk | Hidden lebih kecil, L2/weight decay, atau dropout |
| Loss berosilasi atau tidak stabil | Learning rate lebih kecil atau periksa skala/gradien |
| Train dan validation sama-sama buruk | Audit fitur/target, learning rate, kapasitas, dan epoch |
| Banyak kategori langka | Audit encoding dan kebutuhan penggabungan kategori yang sah |
| Ada riwayat nyata per baterai | Perbandingan T=1,3,5,10 dengan target/horizon konsisten |

Jangan menjalankan kombinasi tambahan tanpa pertanyaan yang hendak dijawab. Dropout/L2 adalah opsional, bukan syarat wajib karena disebut dalam panduan. Jika memakai dropout, pastikan mode evaluasi menonaktifkannya.

## 22. Protokol evaluasi final yang tidak ambigu

1. Selesaikan semua keputusan fitur, preprocessing, konfigurasi, checkpoint, dan seed menggunakan train/validation.
2. Kunci daftar run final dan metrik yang akan dilaporkan.
3. Untuk lingkup tugas ini, gunakan checkpoint model yang dilatih pada train dan dipilih dengan validation. Menggabungkan train+validation untuk retraining tidak wajib dan memerlukan prosedur baru untuk jumlah epoch; jangan melakukannya tanpa dokumentasi.
4. Jalankan evaluasi test untuk baseline dan kandidat final yang sudah dikunci. Bila ada beberapa seed final, evaluasi semuanya dalam tahap final yang sama.
5. Laporkan semua seed; jangan menampilkan hanya yang paling bagus.
6. Setelah melihat test, jangan melanjutkan tuning dengan test yang sama sebagai acuan. Jika perbaikan diperlukan, jelaskan status eksplorasi dan kebutuhan evaluasi baru yang independen.

“Test pada akhir” berarti test tidak ikut memilih model. Bukan berarti hanya boleh ada satu pemanggilan fungsi evaluasi untuk seluruh seed final yang sudah ditetapkan.

Jika pengamatan berulang per baterai, laporkan metrik seluruh baris dan, bila jumlah baterai memadai, ringkasan metrik per baterai. Metrik seluruh baris memberi bobot lebih besar pada baterai dengan lebih banyak pengamatan. Jelaskan perbedaannya.

Standar deviasi antar-seed mengukur variasi training pada split tetap. Itu bukan confidence interval atas populasi baterai dan tidak mencakup seluruh ketidakpastian akibat pemilihan sampel.

Tidak ada target angka universal seperti “akurasi harus 95%” untuk regresi. Keberhasilan proyek dinilai dari metodologi, perbandingan dengan baseline, kualitas analisis, dan kesesuaian error dengan satuan serta tujuan masalah. Jika dosen memberi ambang khusus, catat terpisah.

## 23. Template tabel dan interpretasi hasil

### 23.1 Tabel pemilihan model pada validation

| Run | Input p/T | Hidden | Aktivasi | Loss | Epoch aktual/terbaik | Val MAE | Val RMSE | Parameter | Waktu |
|---|---|---|---|---|---|---|---|---|---|
| B0 | Diisi hasil | | | | | | | | |

Simpan konfigurasi lengkap dalam file hasil meskipun tabel laporan diringkas. Baris dengan run gagal juga dicatat bersama alasan, bukan dibuang tanpa jejak.

### 23.2 Tabel evaluasi final

| Model terkunci | Seed | Test MAE | Test RMSE | Test R² | Catatan |
|---|---|---|---|---|---|
| Baseline konstan | Tidak relevan jika deterministik | | | | Statistik target hanya dari train |
| MLP baseline | | | | | |
| MLP final | | | | | |

Tambahkan rerata ± standar deviasi antar-seed untuk setiap konfigurasi MLP. Baseline konstan hanya pemeriksaan kewajaran tanpa arsitektur pembelajaran lain. Jika dosen melarang semua pembanding di luar MLP secara literal, cukup laporkan baseline MLP sebagai pembanding utama dan konfirmasikan penggunaan baseline konstan dalam laporan.

### 23.3 Struktur paragraf pembahasan

Gunakan urutan:
1. Perubahan yang diuji.
2. Bukti metrik dan kurva.
3. Interpretasi yang masuk akal.
4. Batas kesimpulan.

Template yang harus diisi hasil nyata:
“Pada split yang sama, perubahan [faktor] dari [A] ke [B] mengubah validation MAE dari [hasil] menjadi [hasil]. Kurva menunjukkan [pengamatan]. Temuan ini konsisten dengan [penjelasan], tetapi eksperimen ini belum mengisolasi [keterbatasan].”

Jangan mengisi angka contoh seolah-olah hasil. Kesimpulan “lebih banyak layer selalu lebih baik” tidak sah dari satu dataset dan beberapa konfigurasi.

## 24. Rencana notebook dan fungsi setiap tahap

Nama berikut adalah rencana artefak, bukan file yang sudah dibuat:

### 00_data_audit
- Membaca data asli dan memeriksa skema.
- Mengidentifikasi unit observasi, ID, waktu, target, dan kualitas dasar.
- Menyimpan kamus data serta keputusan awal.
- Menetapkan dan menyimpan manifest split.

### 01_eda_train
- Menggunakan bagian train untuk analisis yang memengaruhi keputusan model.
- Menghasilkan grafik, pemeriksaan redundansi, dan rekomendasi fitur.
- Menulis interpretasi di dekat grafik.

### 02_preprocessing
- Membuat pipeline fit/transform yang dapat digunakan kembali.
- Menyimpan daftar kolom dan parameter preprocessing.
- Memeriksa shape, NaN, keselarasan indeks, serta pemisahan kelompok/waktu.

### 03_mlp_experiments
- Mendefinisikan model dan prosedur training.
- Menjalankan konfigurasi melalui daftar konfigurasi, bukan mengedit angka secara manual tanpa catatan.
- Menyimpan run log, checkpoint, kurva, konfigurasi, dan metrik validation.
- Membuat tabel perbandingan yang tidak memuat tuning dari test.

### 04_final_evaluation
- Memuat konfigurasi/pipeline/checkpoint yang sudah dikunci.
- Menghitung metrik final dan error per kelompok/rentang.
- Membuat grafik aktual-prediksi dan residual.
- Menyimpan ringkasan hasil untuk laporan.

Notebook sebaiknya dapat dijalankan berurutan dari keadaan bersih. Jika tahap tertentu bergantung pada artefak sebelumnya, tulis sumber file dan cara membuatnya. Jangan bergantung pada variabel tersembunyi dari eksekusi sel yang tidak berurutan.

Untuk inferensi baru, artefak minimum adalah model, preprocessing, urutan fitur, definisi target/satuan, dan informasi jendela jika temporal. Bobot model saja tidak cukup.

## 25. Susunan laporan akhir dan bahan presentasi

### Laporan

1. **Pendahuluan:** latar belakang, masalah yang diprediksi, tujuan, batasan MLP/backpropagation.
2. **Landasan teori:** neuron, bobot/bias, MLP, nonlinearitas, loss, chain rule, backpropagation, optimizer, generalisasi. Materi kuliah lain cukup dihubungkan secara singkat.
3. **Dataset:** sumber, versi, asal data yang diketahui, unit observasi, kamus data, target dan skenario prediksi.
4. **EDA:** temuan distribusi, kualitas data, hubungan variabel, dan implikasi keputusan.
5. **Metode:** split, preprocessing, fitur, arsitektur, training, seed, metrik, dan rancangan eksperimen.
6. **Hasil:** baseline, perbandingan eksperimen, kurva, evaluasi final, serta error per rentang/kelompok.
7. **Pembahasan:** alasan hasil, keterbatasan, kemungkinan leakage yang telah diaudit, dan batas penerapan.
8. **Kesimpulan:** jawaban terhadap pertanyaan penelitian berdasarkan hasil aktual, bukan mengulang definisi MLP.
9. **Lampiran:** konfigurasi lengkap, daftar fitur, run log, pembagian kontribusi, dan petunjuk reproduksi.

### Presentasi sekitar 10–12 slide, sesuaikan durasi dosen

- Masalah dan tujuan.
- Dataset, target, dan skenario prediksi.
- Dua atau tiga temuan EDA terpenting.
- Split dan preprocessing beserta alasan.
- Diagram arsitektur MLP.
- Alur backpropagation dan contoh update singkat.
- Rancangan eksperimen.
- Perbandingan hasil validation.
- Hasil test final dan analisis error.
- Keterbatasan dan kesimpulan.

Tidak semua grafik notebook perlu masuk slide. Pilih grafik yang mendukung keputusan atau kesimpulan utama.

## 26. Pembagian kerja operasional dan peninjauan silang

| Paket | Pekerjaan utama | Definisi selesai | Peninjau |
|---|---|---|---|
| A: Data/EDA | Audit, kamus, grafik, asal data | Temuan dapat ditelusuri ke data dan definisi kolom | Anggota preprocessing |
| B: Preprocessing | Split, pipeline, fitur, audit leakage | Pipeline menghasilkan input sah dan tersimpan | Anggota modeling |
| C: Modeling | MLP, training, eksperimen, checkpoint | Run log lengkap dan baseline dapat diulang | Anggota evaluasi |
| D: Evaluasi/laporan | Metrik, error, pembahasan, slide | Semua angka sesuai artefak dan klaim sesuai bukti | Anggota data + modeling |

Jika anggota kurang dari empat, gabungkan peran; jika lebih, bagi eksperimen berdasarkan faktor. Hindari menjalankan eksperimen dengan versi split berbeda. Tetapkan satu pengelola konfigurasi bersama dan satu pengelola tabel hasil.

Titik rapat kelompok:
1. Setelah audit: sepakati target, unit observasi, fitur terlarang, dan tujuan generalisasi.
2. Setelah baseline: pastikan semua anggota memahami dimensi jaringan dan alur backpropagation.
3. Setelah eksperimen: sepakati kandidat final menggunakan validation.
4. Sebelum evaluasi test: kunci keputusan dan cek kelengkapan run log.
5. Sebelum presentasi: cocokkan angka laporan, slide, dan hasil tersimpan.

Urutan ini bukan tenggat kalender. Isi tanggal, penanggung jawab, dan status setiap titik sesuai jadwal kelompok.

## 27. Risiko, gejala, dan tindakan

| Gejala | Kemungkinan yang diperiksa | Tindakan |
|---|---|---|
| Error hampir nol sejak awal | Target tersalin, fitur turunan target, split tidak independen | Audit kolom, rumus target, ID, dan waktu |
| Training loss NaN/tak hingga | Input invalid, learning rate terlalu besar, hitungan numerik | Periksa input, gradien, loss, dan skala |
| Loss tidak berubah | Gradien/optimizer tidak berjalan, aktivasi jenuh, learning rate tidak cocok | Cek satu batch dan perubahan bobot |
| Train sangat baik, validation buruk | Overfitting, distribution shift, kelompok berbeda | Audit split dan kapasitas; pertimbangkan regularisasi |
| Semua model buruk | Target sulit, fitur tidak relevan, bug pipeline, data terbatas | Bandingkan baseline dan periksa X-y |
| Hasil berubah drastis antar-run | Seed, split, inisialisasi, atau lingkungan tidak dikontrol | Kunci split dan catat semua sumber acak |
| MAE bagus tetapi kondisi SOH rendah buruk | Sampel wilayah tersebut sedikit atau pola berbeda | Laporkan metrik per rentang beserta jumlah sampel |
| Timestep panjang terlihat unggul | Sampel evaluasi/horizon berubah | Bandingkan timestamp target dan horizon yang sama |

Tabel ini berisi hipotesis diagnosis, bukan bukti bahwa gejala pasti disebabkan satu hal.

## 28. Pertanyaan yang harus bisa dijawab semua anggota

1. Mengapa masalah ini regresi atau klasifikasi?
2. Apa tepatnya target dan kapan target itu diketahui?
3. Mengapa kolom tertentu masuk atau dikeluarkan?
4. Apa bedanya SOH dengan SOC dan sisa umur baterai?
5. Mengapa neuron input berjumlah p atau T×p?
6. Apa manfaat hidden layer dan aktivasi nonlinear?
7. Apa perbedaan MLP, backpropagation, dan optimizer?
8. Bagaimana gradien dari output mencapai hidden layer?
9. Mengapa bobot diperbarui dengan mengurangi gradien?
10. Apa beda epoch, batch size, dan learning rate?
11. Mengapa output linear dan loss tersebut sesuai target?
12. Mengapa scaler dan imputer hanya fit pada train?
13. Mengapa split tidak selalu boleh acak per baris?
14. Apakah balancing diperlukan dan apa alasannya?
15. Bagaimana model terbaik dipilih tanpa memakai test?
16. Apa yang benar-benar ditunjukkan eksperimen epoch?
17. Apakah model lebih baik daripada baseline dan seberapa stabil antar-seed?
18. Di kondisi mana model gagal dan apa batas penerapannya?
19. Bagaimana orang lain menjalankan ulang hasil kelompok?
20. Bagian apa yang belum dapat disimpulkan dari dataset ini?

## 29. Tingkat prioritas pekerjaan

### Wajib untuk memenuhi ruang lingkup
- Audit data, target sah, EDA berinterpretasi, split tepat, preprocessing beralasan.
- MLP backpropagation dengan penjelasan teori dan diagram dimensi.
- Eksperimen input, hidden layer, aktivasi, loss sesuai masalah, dan epoch.
- Metrik yang benar, validation untuk seleksi, test final, serta pembahasan.
- Hasil tersimpan dan dapat direproduksi.

### Sangat disarankan agar hasil lebih kuat
- Sedikitnya tiga seed final, pencatatan waktu dan parameter.
- Analisis error per rentang target/kelompok.
- Contoh hitung manual backpropagation dan peninjauan silang anggota.
- Pemeriksaan baseline konstan jika sesuai ketentuan dosen.

### Bersyarat atau tambahan
- Timesteps: hanya jika struktur urutan nyata tersedia.
- Balancing: hanya jika jenis target dan ketimpangan memang membutuhkannya.
- Cross-validation: jika ukuran data dan waktu komputasi mendukung serta aturan split dapat dijaga.
- Implementasi dari nol: mengikuti ketentuan dosen.
- Dropout, L2, pencarian learning rate, interpretasi fitur: setelah kebutuhan utama selesai dan ada pertanyaan jelas.

Website, aplikasi deployment, dashboard, model selain MLP, atau klasifikasi tambahan dari SOH bukan syarat proyek berdasarkan instruksi yang tersedia. Fokuskan waktu pada eksperimen dan pemahaman JST terlebih dahulu.

## 30. Checklist pemeriksaan akhir sebelum pengumpulan

### Integritas data dan metode
- [ ] Tidak ada nama kolom, ukuran data, atau satuan yang masih ditebak dalam laporan final.
- [ ] Lisensi dan petunjuk memperoleh data dicantumkan.
- [ ] Seluruh keputusan BELUM DIVERIFIKASI/BELUM DIPUTUSKAN yang relevan sudah diselesaikan.
- [ ] Split dapat direkonstruksi dan tidak melanggar kelompok/waktu yang ditetapkan.
- [ ] Fitur dan preprocessing tidak mengambil informasi test atau masa depan.

### Integritas eksperimen
- [ ] Diagram dan jumlah parameter sesuai model yang benar-benar dijalankan.
- [ ] Semua model menggunakan MLP dengan gradien backpropagation.
- [ ] Konfigurasi, seed, epoch aktual, checkpoint, dan waktu run tercatat.
- [ ] Metrik pembanding memakai satuan dan subset yang konsisten.
- [ ] Hasil eksperimen yang kurang baik tetap masuk catatan.
- [ ] Tidak ada angka ilustrasi yang disajikan sebagai hasil nyata.

### Komunikasi dan reproduksi
- [ ] Setiap grafik memiliki label, satuan, dan interpretasi.
- [ ] Kesimpulan menjawab pertanyaan penelitian dan menyebut keterbatasan.
- [ ] Notebook bisa dijalankan dari keadaan bersih dengan instruksi yang tersedia.
- [ ] Pipeline dan model yang disimpan saling cocok.
- [ ] Setiap anggota dapat menjelaskan bagian 28 dengan bahasanya sendiri.
- [ ] Laporan, slide, dan tabel hasil menyebut angka final yang sama.

Audit CSV dan lembar keputusan awal sudah tersedia. Tahap berikutnya: pendalaman EDA train, lalu pipeline preprocessing sebelum eksperimen MLP. Keputusan implementasi dari nol/framework masih mengikuti ketentuan dosen.
