# Proyek Analisis Data: E-Commerce Public Dataset (Olist)

Analisis data pesanan marketplace e-commerce Brasil (Olist) periode Januari 2017 - Agustus 2018
untuk menjawab tiga pertanyaan bisnis: kategori produk dengan revenue tertinggi, tren pesanan dan
revenue bulanan, serta dampak keterlambatan pengiriman terhadap kepuasan pelanggan. Analisis
lanjutan: RFM, geospasial, dan clustering manual (zona logistik).

## Dashboard

Dashboard di Streamlit Community Cloud:

**[Dashboard](https://data-analysis-olist-yanardika.streamlit.app/)**

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
- `dashboard/dashboard.py` : kode dashboard Streamlit.
- `requirements.txt` : daftar library untuk dashboard.
- `url.txt` : tautan dashboard.

## Menjalankan Notebook (Google Colab)

Notebook disusun untuk dijalankan di [Google Colab](https://colab.research.google.com/), sehingga
tidak perlu menyiapkan environment sendiri (pandas, seaborn, dan folium sudah tersedia di Colab).

1. Download `notebook.ipnyb` dan `E-commerce-public-dataset.zip`
2. Buka `notebook.ipynb` di Colab (*File > Upload notebook*).
3. Upload `E-commerce-public-dataset.zip` ke panel *Files*
4. Runtime -> Run all.


## Menjalankan Dashboard Secara Lokal (opsional)

Dashboard dapat diakses langsung melalui tautan di atas. Untuk menjalankannya di komputer sendiri,
dari folder `submission`:

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
```

Dashboard terbuka di http://localhost:8501.
