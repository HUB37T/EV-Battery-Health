# Split baterai yang dikunci

`battery_split.csv` menyimpan satu baris per battery_id dan subsetnya. Gabungkan dengan CSV asli menggunakan battery_id, bukan posisi baris.

- Unit split: baterai, bukan observasi.
- Tujuan evaluasi: estimasi SOH saat ini pada baterai yang tidak terlihat dalam train.
- Seed: 42, NumPy default_rng, ID diurutkan sebelum permutasi.
- Alokasi: 70% train, 15% validation, 15% test berdasarkan jumlah baterai.
- Hash CSV sumber dan manifest: lihat `../../results/audit/provenance.json`.

Proporsi observasi bisa berbeda dari proporsi baterai. Manifest tidak dipilih berdasarkan target atau performa. Script audit menolak versi CSV berbeda dan menolak menimpa manifest yang tidak cocok. Jangan mengganti seed untuk mencari skor lebih bagus.

ID dalam manifest berasal dari dataset sintetis berlisensi CC0 menurut dokumentasi sumber. Data pengukuran mentah tetap disimpan lokal.
