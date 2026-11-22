# Proyek Analisis Data: E-Commerce Public Dataset (Olist)

Analisis data pesanan marketplace e-commerce Brasil (Olist) periode Januari 2017 - Agustus 2018
untuk menjawab tiga pertanyaan bisnis: kategori produk dengan revenue tertinggi, tren pesanan dan
revenue bulanan, serta dampak keterlambatan pengiriman terhadap kepuasan pelanggan. Analisis
lanjutan: RFM, geospasial, dan clustering manual (zona logistik).

## Struktur Direktori

```
submission
├───dashboard
| ├───main_data.csv
| └───dashboard.py
├───data
| └───E-commerce-public-dataset.zip
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt
```

- `data/E-commerce-public-dataset.zip` : dataset.
- `notebook.ipynb` : proses analisis lengkap (wrangling, EDA, visualisasi, kesimpulan).
- `dashboard/main_data.csv` : data hasil cleaning.
- `dashboard/dashboard.py` : dashboard Streamlit.
- `url.txt` : tautan dashboard yang sudah di-deploy.

## Setup Environment

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Menjalankan Notebook

Buka `notebook.ipynb` di Google Colab lalu jalankan seluruh cell berurutan. Pastikan variabel
`DATA_DIR` mengarah ke folder `data`. Cell terakhir pada tahap persiapan data menghasilkan
`dashboard/main_data.csv`.

## Menjalankan Dashboard

Dari folder `submission`:

```
streamlit run dashboard/dashboard.py
```

Dashboard terbuka di http://localhost:8501.