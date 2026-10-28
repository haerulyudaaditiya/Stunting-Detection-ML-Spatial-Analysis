# Pemeriksaan acuan regresi untuk Reviewer 2

Tahap pertama sudah dijalankan: memeriksa acuan WHO dan hasil regresi tersimpan.
Naskah Word, model penelitian tersimpan, dan dashboard belum diperbarui pada tahap ini.
Pengujian eksternal Jeneponto belum dijalankan untuk model yang akan dipertahankan.

## Temuan utama

Ada dua koreksi pada pembentukan target regresi:

| Target | Temuan | Records yang berubah |
| --- | --- | ---: |
| HAZ | Koefisien umur 24 bulan pada tabel lama merupakan titik tengah tabel panjang dan tinggi. Acuan yang diperiksa menggunakan tabel tinggi mulai umur 24 bulan. | 3,954 |
| WAZ | Kode lama memakai LMS langsung untuk semua nilai. Prosedur WHO menyesuaikan perhitungan pada nilai di luar ±3 SD. | 22,063 |

Enam XLSX WHO yang tersimpan telah diperiksa hash dan ukurannya terhadap manifest.
Tabel acuan dibangun ulang langsung dari XLSX: 244 baris dan 732 koefisien.
Seluruhnya sama persis dengan tabel koreksi yang sudah tersedia.

Perbedaan koefisien hanya terjadi pada dua baris HAZ umur 24 bulan, untuk laki-laki
dan perempuan. Koefisien WAZ tidak berubah; perbedaannya berasal dari perhitungan
di luar ±3 SD. Perubahan maksimum skor adalah sekitar 0.1146 untuk HAZ dan 0.8230
untuk WAZ. WAZ yang berubah mencakup 21,601 records di bawah 24 bulan dan 462
records pada umur 24 bulan. Membatasi umur saja tidak menyelesaikan persoalan WAZ.

## Dampak terhadap hasil Random Forest

Pengujian berikut menggunakan pembagian acak lima fold yang sama dengan kode utama.
Nilai adalah rata-rata lima fold, bukan metrik gabungan seluruh prediksi.

| Target | RMSE dengan target lama | RMSE dengan target koreksi | Tampilan empat desimal, lama → koreksi |
| --- | ---: | ---: | --- |
| HAZ | 0.000794634118 | 0.000800200092 | 0.0008 → 0.0008 |
| WAZ | 0.000153070727 | 0.000136145097 | 0.0002 → 0.0001 |

R² tetap dibulatkan menjadi 1.0000 untuk kedua target, tetapi nilainya bukan tepat
1. Random Forest tetap peringkat pertama dan CatBoost kedua berdasarkan RMSE pada
perbandingan acak utama dengan target koreksi. Ini bukan pernyataan bahwa urutan
tersebut sama pada pengujian berkelompok; pada grouped WAZ, CatBoost berada di atas RF.

RMSE yang berubah tidak menunjukkan peningkatan model dengan sendirinya karena
target pembandingnya juga berubah.

## Apa yang benar-benar diverifikasi

- Pembentukan target lama dijalankan kembali dari sel 3, 5, dan 9 notebook asli.
- Semua input serta target lama cocok dengan rekonstruksi pipeline yang diperiksa.
- Rumus skalar terpisah digunakan untuk memeriksa target koreksi pada 100,000 records.
- Semua 24 cache regresi diperiksa: enam algoritme × dua target × dua versi acuan.
- Fingerprint input, target, parameter model, versi paket, serta pembagian lima fold cocok.
- Rata-rata dan simpangan baku cache cocok dengan lima nilai fold masing-masing.
- Seluruh 36 metrik notebook lama cocok pada presisi enam desimal.
- Random Forest benar-benar dilatih ulang untuk verifikasi: 20 fit, yaitu dua acuan ×
  dua target × lima fold. Setiap metrik cocok dengan cache dalam toleransi absolut 1e−10.
- Lima algoritme lain diperiksa cache/protokol/perhitungannya, tanpa pelatihan ulang baru.
- Tujuh tes data yang relevan lulus.
- Hash 59 file yang dilindungi sama sebelum dan sesudah eksekusi kode audit.

Waktu pelatihan historis tidak dapat langsung digabungkan dengan metrik koreksi.
Notebook asli memakai seluruh worker yang tersedia, sedangkan cache/verifikasi
memakai empat worker. Metadata Python historis juga tidak sama dengan lingkungan
pemeriksaan sekarang. Catatan versi dan batas pembandingan ada di verification.json.

## Hubungan dengan komentar tentang kebocoran

Dalam pembagian acak asli, 99.985% records validasi HAZ memiliki kombinasi input
yang sudah muncul di pelatihan; untuk WAZ angkanya 100%. Selain itu, target regresi
memang dihitung dari variabel yang sama dengan inputnya. Skor hampir sempurna
karena itu perlu dijelaskan sebagai pendekatan terhadap perhitungan indeks pada
data ini. Hasil tersebut tidak membuktikan prediksi pertumbuhan masa depan atau
kinerja klinis pada populasi nyata.

Label klasifikasi Stunting asli tidak diubah. Koreksi ini tidak memerlukan
pelabelan ulang klasifikasi dan tidak membatalkan hasil pengujian klasifikasi
yang menggunakan label sumber tersebut.

## Konsekuensi jika target koreksi digunakan pada naskah akhir

1. Perbarui metrik regresi Table VII dan angka yang diulang pada Abstract/hasil,
   terutama RMSE WAZ Random Forest 0.0002 menjadi 0.0001.
2. Gunakan model regresi yang benar-benar dilatih pada target koreksi untuk gambar
   actual-versus-predicted, SHAP regresi, dashboard, dan pengujian eksternal.
3. SHAP klasifikasi dan waterfall tidak perlu dihitung ulang akibat koreksi target
   regresi ini: model klasifikasinya tidak berubah.
4. Jelaskan pembentukan HAZ/WAZ serta batas pengujian acak. Urutan model terbaik
   tetap dikaitkan dengan protokol yang digunakan untuk memilihnya.
5. Sesuaikan angka dalam respons reviewer yang mengutip metrik lama setelah hasil
   akhir regresi ditetapkan. Jangan menggabungkan metrik target koreksi dengan model lama.

Ini masih hasil audit untuk dibahas bersama penulis. Model regresi akhir belum
diganti, gambar belum diganti, dan angka di Word belum ditulis ulang oleh agen.

## Batas acuan

Perhitungan ini menggunakan umur bulanan serta asumsi panjang di bawah 24 bulan
dan tinggi mulai 24 bulan. Metadata posisi pengukuran dan umur tepat dalam hari
tidak tersedia. Karena itu, acuan tersebut tidak disebut sebagai penilaian klinis
individual yang pasti. Ketidakpastian umur 24 bulan harus dijelaskan tersendiri.

Referensi primer:

- [WHO length/height-for-age](https://www.who.int/tools/child-growth-standards/standards/length-height-for-age)
- [WHO weight-for-age](https://www.who.int/tools/child-growth-standards/standards/weight-for-age)
- [Implementasi perhitungan WHO Anthro](https://github.com/WorldHealthOrganization/anthro/blob/master/R/z-score-helper.R)

## Berkas bukti dan reproduksi

- [Kode audit](../../analysis/revision/reviewer2_reference_audit.py)
- [Perbedaan koefisien](reference_differences.csv)
- [Jumlah target yang berubah](target_changes.csv)
- [Metrik enam algoritme dengan dua acuan](regression_metrics.csv)
- [20 hasil fold RF yang dihitung ulang](fresh_rf_verification.csv)
- [Tumpang tindih input pada pembagian acak](random_fold_overlap.csv)
- [Konfigurasi, pemeriksaan, dan hash sumber](verification.json)

Dari direktori utama repositori:

```sh
python -m analysis.revision.reviewer2_reference_audit
python -m unittest discover -s tests -p 'test_revision_data.py' -v
```

Catatan pelaksanaan: pemeriksaan silang terpisah meregenerasi dua JSON audit
turunan sintetis melalui loader yang ada. Loader tersebut deterministik, tetapi
hash sebelum penulisan ulang kedua JSON tidak direkam sehingga kesamaan byte
sebelum/sesudah tidak diklaim. CSV sumber, tabel WHO, notebook utama, dan model
tersimpan yang dilindungi telah diverifikasi tidak berubah.
