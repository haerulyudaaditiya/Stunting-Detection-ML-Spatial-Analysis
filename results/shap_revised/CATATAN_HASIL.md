# Hasil SHAP Random Forest untuk komentar 5

Kode: [Stunting_SHAP_Analysis_Revised.ipynb](../../Stunting_SHAP_Analysis_Revised.ipynb).
Notebook sudah dijalankan, termasuk tujuh sel kode beserta keluaran aktualnya.
Untuk mengulang, buka dari direktori utama repositori dan jalankan semua sel.

Analisis ini memakai tiga model Random Forest yang tersimpan dari eksperimen
utama: HAZ, WAZ, dan klasifikasi. Model tersebut merupakan hasil pelatihan pada
80% data untuk pengujian holdout, dengan konfigurasi yang dibandingkan dalam
cross-validation utama. SHAP ini tidak merangkum model dari seluruh fold CV.
Model tidak dilatih ulang. Hasil grouped evaluation dan model dari
`results/revision_1/supervised` tidak digunakan dalam analisis ini.

## Hasil

Urutan berikut dihitung dari rata-rata nilai absolut SHAP pada 500 sampel per
tugas. Setiap sampel telah diperiksa berada dalam data uji lama.

| Tugas | Urutan kontribusi dari terbesar |
| --- | --- |
| HAZ | Tinggi badan, umur, jenis kelamin |
| WAZ | Berat badan, umur, jenis kelamin |
| Probabilitas kelas Stunted | Tinggi badan, umur, jenis kelamin, berat badan |

Grafik klasifikasi menjelaskan kelas **Stunted**, bukan rangkuman kepentingan
fitur untuk keempat kelas sekaligus. Nilainya memakai skala probabilitas.
SHAP regresi memakai skala keluaran Z-score masing-masing model.

Waterfall menggunakan anak pertama dalam urutan data uji yang diprediksi
Stunted oleh Random Forest, sesuai aturan pemilihan lama. Indeks baris CSV:
50074; perempuan, 18 bulan, 74.1 cm, 14.3 kg; label asli Stunted.
Nilai dasar 0.2496607291 ditambah kontribusi fitur menghasilkan probabilitas
Stunted sebesar 1.0. Pada contoh ini, kontribusi terbesar adalah umur, diikuti
tinggi badan, jenis kelamin, dan berat badan. Nilai dasar berasal dari model
dengan pembobotan kelas; angka tersebut bukan proporsi kelas dalam dataset.
Probabilitas model ini juga bukan bukti kepastian klinis.

## Gambar untuk diperiksa

- [SHAP regresi HAZ dan WAZ](regression/shap_summary_komparasi.png)
- [SHAP klasifikasi untuk kelas Stunted](classification/classification_shap_summary.png)
- [Waterfall contoh individual](classification/classification_shap_waterfall.png)

Versi PDF tersedia pada direktori yang sama dengan nama dasar yang sama.
Tema, font, susunan panel, warna, ukuran tulisan, dan ekspor 500 DPI mengikuti
kode gambar lama. Ukuran hasil regresi dan klasifikasi summary sama persis
dengan gambar lama. Lebar waterfall berbeda tiga piksel akibat perubahan label
angka saat pemotongan otomatis; gambar tidak diregangkan.

## Bukti pemeriksaan

- [feature_importance.csv](feature_importance.csv): peringkat dan angka SHAP.
- [waterfall_case.csv](waterfall_case.csv): fitur dan kontribusi contoh individual.
- [summary.json](summary.json): model, protokol, skala, kasus, dan keterbatasan.
- [explanations.npz](explanations.npz): indeks baris, input, SHAP, nilai dasar, dan prediksi.
- [verification.json](verification.json): pemeriksaan jumlah kontribusi dan hash sumber.

Nilai dasar ditambah seluruh kontribusi SHAP cocok dengan prediksi model untuk
semua sampel dan seluruh kelas, dengan galat maksimum sekitar 1.6 × 10⁻¹³.
Hash model, data, notebook utama, dan gambar lama tetap sama sesudah eksekusi.

## Penerapan pada paper

Naskah Word belum diubah. Tabel kinerja utama tetap digunakan. Tahap berikutnya
adalah menyesuaikan penyebutan model dan pembahasan SHAP secara manual, termasuk
Abstract, metode, hasil regresi, hasil klasifikasi, contoh waterfall, dan
Conclusion. Caption dan penomoran gambar diselesaikan pada pemeriksaan akhir,
sesuai urutan revisi yang sudah disepakati.

Interpretasi harus mengikuti model dan tugas: WAZ dijelaskan oleh berat badan
dan umur, sedangkan contoh waterfall ini paling banyak dipengaruhi umur.
Hasil SHAP menjelaskan perilaku model pada data sintetis; hasil ini tidak
menetapkan hubungan sebab akibat atau membuktikan validitas klinis.
