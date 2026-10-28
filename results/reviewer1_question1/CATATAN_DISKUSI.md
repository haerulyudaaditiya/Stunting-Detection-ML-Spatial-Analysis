# Catatan diskusi: reviewer 1, komentar 1

Ini catatan pemeriksaan untuk dibahas bersama penulis, bukan teks pengganti naskah atau jawaban final reviewer. Naskah, dashboard, hasil eksperimen terdahulu, notebook asli, dan CSV sumber tidak diubah dalam pemeriksaan ini.

## Pertanyaan yang diuji

Apakah akurasi Random Forest yang hampir sempurna bertahan ketika pengukuran identik dipisahkan antara data latih dan validasi?

## Kontrol eksperimen

- Semua 100.000 baris dan empat label sumber dipertahankan.
- Input tetap jenis kelamin, umur, tinggi, dan berat; HAZ, WAZ, dan Wasting tidak menjadi input.
- Random Forest tetap 50 pohon, class_weight=balanced, random_state model=42. Jumlah worker CPU dibatasi menjadi 4.
- Masing-masing dari tiga cara pembagian diuji dengan lima fold dan seed pembagian 42, 43, 44. Total 45 pelatihan. Seed model tetap 42 di seluruh percobaan.
- Semua pengujian baru memakai lingkungan software yang sama. Versi eksekusi notebook historis belum diketahui lengkap; reproduksi kode pada lingkungan saat ini tidak membuktikan kesamaan versi historis.
- Nilai utama dihitung sebagai rata-rata lima fold, sesuai notebook asli, lalu dirata-ratakan pada tiga seed. Metrik gabungan seluruh prediksi out-of-fold juga disimpan.
- Rentang antar-seed adalah pemeriksaan sensitivitas terhadap pembagian, bukan interval kepercayaan atau uji signifikansi. Tidak ada pemilihan model atau seed berdasarkan hasil ini.

## Hasil Random Forest

| Pembagian | Akurasi rata-rata tiga seed | Rentang akurasi antar-seed | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Acak berstrata (metode awal) | 99.973% | 99.967–99.978% | 0.9994 | 99.903% |
| Kelompok empat input identik | 99.971% | 99.967–99.975% | 0.9993 | 99.895% |
| Kelompok jenis kelamin–umur–tinggi | 97.828% | 97.801–97.875% | 0.9550 | 94.691% |

Pada seed 42, pembagian acak menghasilkan akurasi 99,978%, konsisten dengan pembulatan 99,98% pada notebook. Pengelompokan jenis kelamin–umur–tinggi pada seed yang sama menghasilkan 97,809%.

### Recall per kelas pada pengelompokan jenis kelamin–umur–tinggi

Angka berikut adalah rata-rata fold dan tiga seed, bukan sensitivitas klinis pada populasi nyata.

| Kelas | Recall |
|---|---:|
| Normal | 99.26% |
| Severely Stunted | 93.88% |
| Stunted | 95.61% |
| Tall | 90.01% |

## Pemeriksaan penjelas tambahan: pencarian label pada pola yang sama

Pemeriksaan ini dilakukan setelah melihat hasil awal, sehingga diposisikan sebagai diagnostik eksploratif. Pada pembagian acak seed 42, sebuah tabel pencarian menyimpan label dari data latih untuk setiap kombinasi jenis kelamin–umur–tinggi. Jika kombinasi validasi belum terlihat, prediksi memakai kelas mayoritas dari data latih fold itu. Label validasi tidak digunakan untuk mengisi tabel.
Metode sederhana ini mencapai akurasi 99.997%. Dari 100.000 prediksi validasi, hanya 15 berasal dari kombinasi yang belum ada di pelatihan.
Ini menunjukkan bahwa pembagian acak tersebut dapat dilalui dengan sangat baik hanya dengan mengenali kembali pola pengukuran. Ini tidak membuktikan bahwa Random Forest benar-benar hanya menghafal, dan tidak mengungkap rumus asli pembentukan label.

## Penilaian terhadap empat kekhawatiran reviewer

1. **HAZ/WAZ sebagai fitur:** pemeriksaan kode asli menunjukkan keduanya tidak masuk X_clean. Ini bukti tentang jalur input eksplisit; bukan pembuktian bahwa seluruh kemungkinan kebocoran atau dependensi telah tersingkir.
2. **Dominasi kelas:** Normal mencakup 72,312% data. Prediksi konstan Normal mencapai akurasi 72,312%, jauh dari 99,98%. Pada pembagian acak, macro F1 dan recall kelas minoritas juga tinggi. Dominasi kelas saja tidak menjelaskan skor hampir sempurna, tetapi tetap membuat akurasi saja tidak cukup.
3. **Pengulangan input:** pemisahan empat input identik hampir tidak mengubah skor. Penurunan yang lebih jelas terjadi saat kombinasi jenis kelamin–umur–tinggi dipisahkan. Kedua pembagian menguji hal berbeda; pengelompokan tiga input merupakan pengujian lebih ketat terhadap kombinasi pengukuran yang belum terlihat.
4. **Label sintetis mudah dipelajari:** hasil pencarian pola menunjukkan keteraturan dan pengulangan yang cukup untuk skor sangat tinggi pada pembagian acak. Pemeriksaan dokumentasi sumber sebelumnya tidak menemukan rumus atau kode pembentukan label. Kita belum dapat menyatakan bahwa label dibuat persis dengan standar WHO.

## Kesimpulan yang didukung bukti

Angka 99,98% dapat direproduksi melalui pembagian acak dan fitur antropometri eksplisit pada dataset sintetis ini. Namun, hampir seluruh kombinasi jenis kelamin–umur–tinggi pada validasi telah muncul di pelatihan. Skor tersebut tidak cukup untuk menunjukkan kemampuan pada kombinasi pengukuran yang belum terlihat. Pengelompokan kombinasi tersebut menghasilkan akurasi sekitar 97,83% dan macro F1 sekitar 0,955, dengan arah penurunan konsisten pada tiga seed.

Penyebab tunggal skor tinggi tidak dapat dipastikan. Penghapusan tumpang tindih empat input saja tidak cukup menurunkannya; keteraturan label sintetis dan pola tiga input yang berulang tetap merupakan penjelasan yang didukung pemeriksaan. Kita tidak boleh menyebut hasil ini bukti akurasi klinis, independensi antar-anak, atau kemampuan generalisasi nasional.

Pengelompokan berdasarkan pengukuran bukan pengelompokan berdasarkan identitas anak. Data tidak mempunyai ID anak. Kombinasi yang berbeda tetap dapat berdekatan, sehingga pengujian ini juga belum menyingkirkan seluruh dependensi sintetis.

## Bahan untuk evaluasi bersama

- Apakah pembagian acak dipertahankan sebagai hasil pembanding, dengan pengujian berkelompok sebagai bukti tambahan utama?
- Bagian metode perlu menjelaskan empat input, label sumber, unit pengelompokan, dan keterbatasan identitas anak.
- Bagian hasil perlu membedakan akurasi dari macro F1/recall kelas; hindari menyamakan status saat pengukuran dengan prediksi masa depan.
- Belum ada kesimpulan bahwa Random Forest adalah model terbaik setelah perubahan validasi; eksperimen ini sengaja hanya membandingkan Random Forest dengan dirinya sendiri.
- Penulis akan merumuskan perubahan naskah dan jawaban reviewer setelah mengevaluasi temuan ini bersama.

## Berkas dan reproduksi

- `summary.json`: konfigurasi, versi, hash, metrik 45 fold, distribusi kelas, overlap, dan ringkasan tiga seed.
- `comparison.csv`: hasil sembilan percobaan.
- `oof_predictions_and_folds.csv`: prediksi dan fold untuk setiap baris pada sembilan percobaan.
- `lookup_diagnostic.json` dan `reproduce_lookup.py`: diagnostik pencarian pola yang dijelaskan di atas.
- `verification.json`: pemeriksaan ulang metrik, overlap, dan keutuhan sumber.

Dari root repository:

```sh
python -m analysis.revision.reviewer1_splits
python results/reviewer1_question1/reproduce_lookup.py
python -m unittest discover -s tests -v
```

Deskripsi sumber dataset: https://www.kaggle.com/datasets/jabirmuktabir/stunting-wasting-dataset
