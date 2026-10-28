# Hasil pemeriksaan dan pengujian untuk Reviewer 2

Eksperimen dijalankan pada 18 Oktober 2026. Tahap komputasi dan pemeriksaan hasil
selesai; penyesuaian naskah dan jawaban reviewer masih perlu dikerjakan bersama
penulis. Word, eksperimen lama, dan dashboard tidak diubah pada tahap ini.

## Apa yang dikerjakan

1. Menggunakan target HAZ/WAZ yang sudah dikoreksi dan diaudit terhadap tabel WHO.
   HAZ umur 24 bulan memakai tabel tinggi badan, bukan rata-rata tabel panjang
   dan tinggi. WAZ menerapkan ekstensi perhitungan di luar ±3 SD. Sebanyak 3954
   nilai HAZ dan 22063 nilai WAZ berubah; input serta label klasifikasi tetap.
2. Menjalankan ulang enam algoritme regresi untuk dua target dengan CV acak dan
   CV berkelompok: 24 perbandingan, masing-masing lima fold, **120 pelatihan baru**.
   Hasil lama tidak disalin sebagai keluaran pelatihan baru.
3. Melatih 12 model regresi pada pembagian 80:20 yang sama dengan eksperimen
   utama. RF dipilih untuk HAZ/WAZ berdasarkan CV acak utama. RF klasifikasi
   asli tetap digunakan. Seluruh model terpilih dilatih pada 80000 baris sintetis.
4. Mencatat hash ketiga model, data, acuan, serta indeks pembagian sebelum
   pengujian Jeneponto. Jeneponto tidak digunakan untuk melatih, memilih model,
   atau mengubah aturan keputusan.
5. Menghasilkan ulang dua gambar actual-versus-predicted dan satu gambar gabungan
   SHAP regresi menggunakan model koreksi. Gaya dan ukuran gambar asli dipertahankan.
6. Menguji ketiga model pada Jeneponto, kemudian memeriksa ulang seluruh prediksi
   dan metrik dari model tersimpan melalui perhitungan terpisah.

## Hasil utama pada Jeneponto

Umur **0–23 bulan** menjadi pengujian utama, sesuai protokol sebelum pengujian.
Umur **24 bulan** dilaporkan terpisah karena transisi acuan panjang/tinggi dan
ketidaktersediaan posisi pengukuran. Pemisahan ini tidak menghilangkan seluruh
keterbatasan pengukuran pada umur lainnya.

`n` menyatakan jumlah catatan yang layak untuk masing-masing tugas, bukan jumlah
anak unik yang telah teridentifikasi.

| Kelompok | Tugas | n | Hasil |
| --- | --- | ---: | --- |
| 0–23 bulan | Klasifikasi biner | 9755 | Akurasi 92.27%; sensitivitas 85.43%; spesifisitas 94.56% |
| 0–23 bulan | Regresi HAZ | 9760 | MAE 0.0323; RMSE 0.1913; R² 0.9814 |
| 0–23 bulan | Regresi WAZ | 9758 | MAE 0.0006; RMSE 0.0168; R² 0.9997 |
| 24 bulan | Klasifikasi biner | 666 | Akurasi 78.53%; sensitivitas 100.00%; spesifisitas 68.85% |
| 24 bulan | Regresi HAZ | 666 | MAE 0.0316; RMSE 0.1875; R² 0.9778 |
| 24 bulan | Regresi WAZ | 666 | MAE 0.0014; RMSE 0.0193; R² 0.9996 |

Klasifikasi mempertahankan keputusan empat kelas model, kemudian menggabungkan
Severely Stunted/Stunted sebagai positif dan Normal/Tall sebagai negatif. Jumlah
probabilitas kelas positif hanya digunakan untuk AUC, bukan membuat ambang baru.
Label pembanding adalah label biner sumber. Akurasi eksternal biner tidak identik
protokolnya dengan angka 99.98% empat kelas pada CV sintetis utama.

Pada kelompok utama, TP=2088, FN=356, TN=6913, FP=398. Jadi hasil ini masih
melewatkan 356 dari 2444 catatan yang diberi label stunting oleh sumber.
Macro-F1=0.8977; balanced accuracy=0.8999; AUC=0.9416.

Interval bootstrap profil 95% untuk akurasi utama adalah 90.75–93.66%; untuk
umur 24 bulan 69.17–87.76%. Bootstrap menggunakan 2000 pengulangan, dengan kelompok
jenis kelamin–umur–tinggi untuk klasifikasi dan tiga input masing-masing regresor.
Ini bukan interval berdasarkan identitas anak unik; ketergantungan antar keluarga,
fasilitas, atau kunjungan tidak dapat diperiksa dari metadata yang tersedia.

## Mengapa skor regresi tetap tinggi perlu dibatasi

Target regresi dihitung dari variabel yang sama dengan input model. Pengujian
regresi Jeneponto menguji pendekatan terhadap perhitungan WHO bulanan pada
pengukuran nyata; targetnya bukan outcome klinis yang diamati secara independen.

Selain itu, kombinasi input eksternal sering sudah tersedia pada **80000 baris
pelatihan sebenarnya**:

| Tugas/input | Catatan utama dengan kombinasi input yang ada di pelatihan |
| --- | ---: |
| HAZ: jenis kelamin, umur, tinggi | 9313/9760 (95.42%) |
| WAZ: jenis kelamin, umur, berat | 9735/9758 (99.76%) |
| Klasifikasi: seluruh empat input | 1010/9755 (10.35%) |
| Klasifikasi: jenis kelamin, umur, tinggi saja | 9308/9755 (95.42%) |

Kesamaan profil pengukuran tidak membuktikan kesamaan anak atau duplikasi
antar sumber. Namun, hasil ini juga tidak boleh disebut pengujian yang seluruh
kombinasi inputnya baru. CV berkelompok menyediakan pemeriksaan tambahan ketika
kombinasi input identik benar-benar dipisahkan.

Untuk RF dengan target koreksi:

| Target | RMSE CV acak | RMSE CV berkelompok |
| --- | ---: | ---: |
| HAZ | 0.00080020 | 0.04182848 |
| WAZ | 0.00013615 | 0.09943635 |

RF tetap pertama pada CV acak kedua target. Pada CV berkelompok HAZ, RF juga
pertama. Pada CV berkelompok WAZ, CatBoost terbaik (RMSE 0.0281302857), sedangkan
RF peringkat keempat. Kita tidak mengganti pilihan model berdasarkan hasil
Jeneponto. Klaim RF terbaik harus tetap menyebut konteks perbandingan acak utama.

## Acuan data dan batas kebocoran

- Fitur model klasifikasi hanya jenis kelamin, umur, tinggi, dan berat; HAZ/WAZ
  tidak dimasukkan. Label Stunting sintetis asli dipertahankan.
- Dalam CV acak regresi, 99.985% catatan validasi HAZ dan 100% WAZ mempunyai
  kombinasi input yang juga ada di fold pelatihan. Dalam CV berkelompok, overlap
  kombinasi input tersebut nol. Ini membantu menjelaskan skor acak yang tinggi.
- Kode pembangkit data sintetis dan identitas anak tidak tersedia. Pemeriksaan
  ini tidak membuktikan bahwa seluruh bentuk kebocoran atau ketergantungan data
  telah dikesampingkan. Hindari pernyataan mutlak bahwa tidak ada kebocoran.
- Jeneponto berasal dari sumber pengukuran nyata yang berbeda, tetapi pernah
  digunakan dalam evaluasi eksploratoris dengan artefak model lain sebelumnya.
  Pengujian sekarang menggunakan model RF yang telah ditentukan dari perbandingan
  sintetis dan dibekukan; ini bukan kohor prospektif yang belum pernah dilihat.

XLSX sumber berisi 31740 catatan. Pembacaan ulang cocok dengan seluruh 29 kolom
CSV harmonisasi. Ada 20158 sel berat berformat tanggal dengan format angka
eksplisit `d\.m`; nilainya dipulihkan dari representasi tampilan Excel, bukan
diestimasi. Sebanyak 6445 catatan klasifikasi utama memakai pemulihan ini.
Profil berulang dipertahankan karena tidak ada identitas anak untuk membuktikan
bahwa catatan tersebut merupakan anak atau kunjungan yang sama.

Ada 430 ketidaksesuaian antara label biner sumber dan kategori HAZ bulanan dari
9760 catatan utama yang dapat dibandingkan. Ketidaksesuaian tidak dipakai untuk
menghapus atau mengganti label. Posisi pengukuran dan umur tepat dalam hari tidak
tersedia. Jeneponto adalah satu wilayah dan tidak mewakili seluruh Indonesia.
Hasil tidak membuktikan validitas klinis, prediksi pertumbuhan mendatang, atau
kesiapan menggantikan penilaian tenaga kesehatan.

## Dampak pada naskah yang perlu dibahas bersama

1. Perbarui angka regresi Table VII beserta angka yang berulang di Abstract dan
   pembahasan. RMSE RF HAZ tetap 0.0008 pada empat desimal; RF WAZ berubah dari
   0.0002 menjadi 0.0001. Metrik algoritme lain juga harus mengikuti hasil koreksi.
   Perubahan RMSE antar target lama/koreksi tidak otomatis berarti model membaik,
   karena angka acuan pembandingnya ikut berubah.
2. Dua gambar actual-versus-predicted dan gambar SHAP regresi memakai hasil baru.
   Urutan kontribusi SHAP tetap: HAZ tinggi–umur–jenis kelamin; WAZ berat–umur–jenis
   kelamin. Gambar SHAP klasifikasi dan waterfall terdahulu tidak berubah.
3. Tambahkan metode dan hasil pengujian eksternal secara ringkas, termasuk hasil
   umur 24 bulan yang lebih rendah serta batas arti regresi dan overlap profil.
4. Sesuaikan klaim kesimpulan dan jawaban Reviewer 2. Jangan menyatakan validasi
   klinis atau keterwakilan nasional. Respons Reviewer 1 yang mengutip RMSE WAZ
   lama juga perlu diperbarui supaya konsisten.
5. Waktu pelatihan di CSV baru benar-benar diukur dalam CV yang baru. Angka lama
   tidak digabungkan dengan metrik baru. Waktu bergantung lingkungan dan beban
   proses; pengukuran ini bukan benchmark perangkat keras yang dikendalikan.
6. Dashboard belum memakai model koreksi ini. Penyelarasan dashboard masih perlu
   dibahas sebelum paket akhir, bukan dianggap sudah dilakukan.

Ini catatan eksperimen untuk evaluasi bersama, bukan teks pengganti otomatis
untuk Word atau jawaban reviewer yang sudah final.

## Berkas dan pemeriksaan

- [Notebook revisi regresi yang telah dieksekusi](../../Stunting_Regression_Analysis_Revised.ipynb)
- [Metrik CV seluruh enam algoritme dan dua protokol](../regression_corrected/cv_metrics.csv)
- [Model yang dibekukan sebelum pengujian eksternal](../regression_corrected/model_freeze_manifest.json)
- [Gambar regresi baru, PNG dan PDF](../regression_corrected/plots/)
- [Kode tambahan pengujian eksternal](../../analysis/revision/reviewer2_external_validation.py)
- [Metrik eksternal lengkap](external_metrics.csv)
- [Audit, protokol, CI, dan keterbatasan](external_validation.json)
- [Pemeriksaan terpisah oleh root](verification.json)
- [Audit acuan WHO sebelumnya](../reviewer2_reference_audit/CATATAN_HASIL.md)

Seluruh 36 tes Python lulus. Notebook valid dengan tujuh sel kode yang telah
dieksekusi, tiga gambar tertanam, dan tanpa keluaran error. Seluruh 187 hash
berkas sumber/artefak lama yang dilindungi tetap cocok. Prediksi keenam kelompok
eksternal dihitung kembali dari model beku dan metrik diverifikasi dengan rumus
terpisah. SHAP memenuhi penjumlahan ke prediksi dengan galat maksimum <2e−13.

Gambar scatter tetap 8920×4920 piksel; SHAP gabungan 7743×2920 piksel; seluruhnya
500 DPI dan telah diperiksa secara visual. Peringatan SHAP tentang perubahan
perilaku RNG pada versi mendatang tidak menghentikan eksekusi saat ini.

Reproduksi dari direktori repositori:

```sh
# Jalankan semua sel notebook revisi regresi melalui Jupyter/VS Code terlebih dahulu.
MPLCONFIGDIR=/tmp/icdees-external-mpl OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m analysis.revision.reviewer2_external_validation \
  --manifest results/regression_corrected/model_freeze_manifest.json \
  --output results/reviewer2_validation

MPLCONFIGDIR=/tmp/icdees-tests-mpl OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m unittest discover -s tests -v
```

Menjalankan ulang notebook dapat memakai cache CV yang cocok dan melatih ulang
model holdout, sehingga manifest/model dan hasil eksternal harus diperbarui
bersama. Jangan mencampur hasil dari hash model berbeda.

Sumber primer:

- [Dataset Jeneponto, Mendeley Data versi 4](https://data.mendeley.com/datasets/wzwpc9j5bx/4)
- [WHO length/height-for-age](https://www.who.int/tools/child-growth-standards/standards/length-height-for-age)
- [WHO weight-for-age](https://www.who.int/tools/child-growth-standards/standards/weight-for-age)
- [Implementasi WHO Anthro](https://github.com/WorldHealthOrganization/anthro/blob/master/R/z-score-helper.R)
