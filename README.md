# Interpretable Machine Learning for Child Stunting Assessment

Kode dan hasil penelitian untuk paper ICDEES 2026, nomor **1571345184**. Penelitian mencakup clustering provinsi, klasifikasi stunting, dan regresi HAZ/WAZ.

## Notebook

| Notebook | Isi |
|---|---|
| `Stunting_Classification_Analysis.ipynb` | Eksperimen klasifikasi lima model |
| `Stunting_Clustering_Analysis.ipynb` | Analisis clustering awal |
| `Stunting_Regression_Analysis.ipynb` | Analisis regresi dengan perhitungan target awal |
| `Stunting_Clustering_Analysis_Revised.ipynb` | Clustering revisi dengan indikator dan agregasi provinsi yang diperbaiki |
| `Stunting_Regression_Analysis_Revised.ipynb` | Regresi revisi dengan target WHO yang diperbaiki dan enam model |
| `Stunting_SHAP_Analysis_Revised.ipynb` | Analisis SHAP klasifikasi; SHAP regresi final ada di notebook regresi revisi |

Notebook lama tetap disimpan karena analisis revisi dan pemeriksaannya masih memakai sebagian isi atau hasilnya. Menjalankan ulang notebook dapat melatih model dan mengganti berkas hasil.

## Folder utama

- `dataset/`: data penelitian, sumber referensi, dan data turunan.
- `analysis/revision/`: skrip audit dan evaluasi tambahan.
- `results/`: model, metrik, prediksi, dan gambar hasil analisis.
- `docs/`: dashboard dan ekspor model untuk browser.
- `tests/`: pemeriksaan kode dan hasil.

## Menjalankan dashboard

Jalankan dari folder utama repo:

```sh
python -m http.server 8000 --directory docs
```

Buka <http://localhost:8000> di browser.

Untuk memasang paket analisis:

```sh
python -m pip install -r requirements.txt
```

## Catatan penelitian

- Model terpilih dalam paper adalah Random Forest. Data Jeneponto digunakan untuk evaluasi eksternal setelah model dibekukan; usia 0–23 bulan menjadi analisis utama dan usia 24 bulan dilaporkan terpisah.
- Regresi HAZ/WAZ memperkirakan nilai dari referensi WHO menggunakan variabel antropometri yang juga menjadi input. Hasilnya bukan prediksi pertumbuhan masa depan atau diagnosis.
- Clustering merangkum indikator kabupaten/kota pada 34 provinsi historis. Hasilnya berupa kelompok deskriptif, bukan prevalensi berbobot populasi.

Jalankan pemeriksaan kode dengan:

```sh
python -m unittest discover -s tests -v
```
