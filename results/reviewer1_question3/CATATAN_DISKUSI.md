# Catatan diskusi — Reviewer 1, komentar 3

Catatan ini memisahkan kesalahan sumber data, hasil perhitungan ulang, dan pilihan metode. Ini bukan naskah revisi atau jawaban final reviewer. Tidak ada notebook asli, workbook sumber, naskah, dashboard, atau hasil lama yang diubah.

## Apa yang reviewer minta

Reviewer meminta agar Silhouette 0,3199 diinterpretasikan sebagai pemisahan yang terbatas dan clustering diposisikan sebagai eksplorasi. Reviewer tidak secara khusus memerintahkan perubahan algoritma atau jumlah cluster.

## Kesalahan yang terkonfirmasi

1. Excel kolom E berada di bawah JUMLAH BALITA / SSGBI 2019 dengan SUPAS 2021 / Stunting. Kode asli menamainya Total_Births. Ini jumlah perkiraan balita stunting, bukan jumlah kelahiran.
2. Kode menggunakan jumlah tersebut sebagai bobot keempat indikator provinsi. Bobot itu bukan denominator yang sesuai untuk seluruh indikator yang mencakup populasi berbeda, seperti ibu hamil dan anak usia imunisasi.
3. Kolom tersebut juga masuk sebagai fitur kelima. Memasukkan indikator jumlah/beban bukan selalu salah, tetapi maknanya harus benar dan sesuai tujuan. Untuk profil empat indikator proporsi, mengeluarkannya adalah pilihan yang sederhana.
4. Pada tabel profil lama, KEK, imunisasi, dan BBLR masih berupa proporsi, tetapi diberi label persen. Misalnya 0,16 merupakan sekitar 16%, bukan 0,16%.
5. Label Moderate/Critical/Low ditetapkan dari nomor cluster. Nomor cluster tidak mempunyai makna risiko klinis bawaan. Label tersebut memerlukan definisi serta validasi yang tidak tersedia dalam analisis ini.

## Hal yang bukan penyebab perubahan hasil

KNN imputation pada data provinsi lama tidak mengubah satu nilai pun: seluruh ringkasan provinsi sudah terisi. Menghapus langkah ini adalah penyederhanaan, bukan penyebab perubahan skor.

## Pilihan koreksi yang diperiksa

Analisis koreksi menggunakan empat indikator (stunting, KEK, imunisasi, BBLR), dengan rata-rata kabupaten/kota yang tersedia dalam setiap provinsi. Tidak ada denominator populasi yang ditebak. Ini pilihan ringkasan deskriptif yang dapat dijelaskan, bukan satu-satunya metode agregasi yang mungkin dan bukan prevalensi provinsi berbobot penduduk.

Data tetap 514 kabupaten/kota dan 34 provinsi historis. Nilai yang hilang pada tingkat kabupaten/kota: stunting 0, KEK 61, imunisasi 11, BBLR 40. Cakupan indikator berbeda-beda; seluruh provinsi masih memiliki pengamatan untuk setiap indikator. Terdapat 430 baris kabupaten/kota lengkap untuk empat indikator.

## Perbandingan terkontrol dengan pengaturan model asli

Pengaturan constructor model mengikuti notebook asli pada lingkungan saat ini: KMeans dan GMM dengan random_state=42, dan Hierarchical menggunakan Ward. Fitur dan agregasi diperbaiki, kemudian k=3 dipertahankan terlebih dahulu. Kasus k=2 ditampilkan terpisah. Versi historis pustaka notebook belum lengkap diketahui; hasil rekonstruksi di lingkungan sekarang cocok dengan angka asli yang tersimpan.

| Skenario | Algoritma | k | Silhouette | DBI | CHI |
|---|---|---:|---:|---:|---:|
| Kode asli direkonstruksi | K-Means | 3 | 0.2182 | 1.2721 | 12.10 |
| Kode asli direkonstruksi | GMM | 3 | 0.1995 | 1.3236 | 11.34 |
| Kode asli direkonstruksi | Ward | 3 | 0.3199 | 0.9877 | 12.38 |
| Pengolahan dikoreksi, k tetap 3 | K-Means | 3 | 0.2586 | 1.1703 | 18.00 |
| Pengolahan dikoreksi, k tetap 3 | GMM | 3 | 0.2449 | 1.1941 | 17.10 |
| Pengolahan dikoreksi, k tetap 3 | Ward | 3 | 0.2746 | 1.1656 | 18.30 |
| Pengolahan dikoreksi, k diubah ke 2 | K-Means | 2 | 0.2800 | 1.2066 | 17.03 |
| Pengolahan dikoreksi, k diubah ke 2 | GMM | 2 | 0.3464 | 1.1307 | 18.96 |
| Pengolahan dikoreksi, k diubah ke 2 | Ward | 2 | 0.3852 | 0.9358 | 17.20 |

Pada k=3 dan pengaturan model asli, Ward memiliki Silhouette dan CHI tertinggi serta DBI terendah relatif terhadap dua pembanding, tetapi Silhouette turun menjadi 0,2746. Perbaikan pengolahan tidak otomatis meningkatkan skor.

Angka Ward 0,3852 diperoleh pada k=2. Perubahan dari 0,3199 menjadi 0,3852 tidak boleh disebut akibat memperbaiki kolom saja karena jumlah cluster juga berubah.

## Mengapa hasil koreksi sebelumnya perlu dibedakan

Skrip analysis/revision/clustering.py juga menggunakan KMeans n_init=50 serta GMM n_init=20 dan max_iter=1000. Ini membantu memeriksa beberapa inisialisasi tetapi merupakan perubahan konfigurasi tambahan, bukan tuntutan langsung reviewer.

Pada pengaturan tersebut, hasil k=3 adalah KMeans 0,2921; GMM 0,2933; Ward 0,2746. Karena itu peringkatnya berbeda dari hasil terkontrol yang mempertahankan constructor asli. Jangan mencampurkan kedua konfigurasi atau membawa klaim Ward terbaik pada semua metrik ke setiap skenario.

Hasil profil, seluruh enam baris metrik koreksi k=2/k=3, dan kurva pemilihan k dalam results/revision_1/clustering telah dihitung ulang secara independen dan cocok dalam toleransi 1e-10.

## Pilihan jumlah cluster untuk dibahas bersama

- **Tetap k=3:** menjaga jumlah kelompok dan konfigurasi model seperti eksperimen awal. Dengan agregasi yang dikoreksi, skor Ward menjadi 0,2746. Tiga kelompok harus dijelaskan sebagai pembagian deskriptif yang dipilih untuk eksplorasi, bukan jumlah optimal atau tingkatan risiko klinis tervalidasi.
- **Gunakan k=2:** mengikuti maksimum Silhouette KMeans pada k=2 sampai 8 di data koreksi. k=2 tertinggi baik dengan constructor asli maupun n_init=50, walaupun nilainya berbeda. Keputusan ini mengubah jumlah kelompok, anggota, tabel, dan gambar. Pemilihan k dari KMeans tidak membuktikan k=2 optimal untuk semua algoritma.

Naskah awal sendiri sudah menyatakan bahwa Silhouette KMeans tertinggi pada k=2 tetapi memilih k=3 demi tiga kategori risiko. Jadi, memilih k=2 bukan penemuan bahwa koreksi kolom otomatis mengubah k terbaik; ini juga perubahan dasar pemilihan jumlah kelompok.

## Kesimpulan untuk evaluasi bersama

Kesalahan penamaan dan penggunaan kolom perlu dikoreksi. Penggunaan empat indikator dengan rata-rata kabupaten/kota adalah pilihan metode yang harus dinyatakan secara terbuka. Mengubah jumlah cluster dan pengaturan inisialisasi merupakan keputusan terpisah. Semua hasil tetap perlu diposisikan sebagai eksploratif; skor internal tidak membuktikan kategori risiko klinis atau efektivitas intervensi.

Dalam diskusi lanjutan pada 18 Oktober 2026, penulis menyetujui dua kelompok berdasarkan pemilihan Silhouette KMeans, dengan constructor model asli dipertahankan. Versi yang telah dijalankan adalah [Stunting_Clustering_Analysis_Revised.ipynb](../../Stunting_Clustering_Analysis_Revised.ipynb), dengan keluaran di [results/clustering_revised](../clustering_revised/README.md). Pada konfigurasi ini Ward unggul pada Silhouette/DBI, sedangkan GMM unggul pada CHI. Naskah masih akan diperbarui secara manual oleh penulis.

## Bukti dan cara mengulang

- analysis.json: hash sumber, versi, sembilan hasil terkontrol, kurva k dan hasil pemeriksaan.
- controlled_comparison.csv: tabel hasil terkontrol di atas.
- province_summary_comparison.csv: perbandingan empat indikator untuk setiap provinsi dalam satuan persen.
- reproduce_analysis.py: rekonstruksi bagian kode asli dan pemeriksaan koreksi tanpa menimpa sumber.

```sh
MPLCONFIGDIR=/tmp/icdees-mpl python results/reviewer1_question3/reproduce_analysis.py
MPLCONFIGDIR=/tmp/icdees-mpl python -m unittest discover -s tests -p test_revision_clustering.py -v
```

Sembilan tes clustering lulus. Notebook dan workbook asli tetap utuh berdasarkan checksum.
