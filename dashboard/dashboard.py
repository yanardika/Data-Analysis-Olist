from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# set_page_config wajib menjadi perintah Streamlit pertama
st.set_page_config(page_title="Dashboard E-Commerce Olist", layout="wide")
sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).resolve().parent / "main_data.csv"

# Hanya kolom yang dipakai dashboard yang dibaca (hemat memori di Streamlit Cloud)
USE_COLS = [
    "order_id", "order_item_id", "price", "order_purchase_timestamp",
    "customer_unique_id", "customer_state", "review_score",
    "delivery_days", "delay_days", "product_category_name_english",
]

# Titik pusat (rata-rata lintang/bujur) tiap state, dihitung dari geolocation_dataset.csv
STATE_COORDS = {
    "AC": (-9.709, -68.453), "AL": (-9.608, -36.081), "AM": (-3.347, -60.536),
    "AP": (0.074, -51.204), "BA": (-13.067, -39.63), "CE": (-4.395, -39.028),
    "DF": (-15.814, -47.98), "ES": (-20.085, -40.507), "GO": (-16.567, -49.344),
    "MA": (-3.821, -44.818), "MG": (-19.858, -44.448), "MS": (-20.791, -54.535),
    "MT": (-14.22, -55.72), "PA": (-2.694, -49.554), "PB": (-7.094, -35.864),
    "PE": (-8.19, -35.789), "PI": (-5.7, -42.487), "PR": (-24.765, -50.953),
    "RJ": (-22.712, -43.152), "RN": (-5.851, -36.01), "RO": (-10.358, -62.708),
    "RR": (2.729, -60.678), "RS": (-29.65, -52.068), "SC": (-27.222, -49.636),
    "SE": (-10.85, -37.2), "SP": (-23.087, -47.162), "TO": (-9.491, -48.358),
}
DELAY_LABELS = ["Tepat waktu / lebih cepat", "Terlambat <= 3 hari", "Terlambat 3-7 hari", "Terlambat > 7 hari"]
DELAY_COLORS = dict(zip(DELAY_LABELS, ["#5BA37C", "#E6B422", "#E08E2B", "#D9534F"]))
SEGMENTS = ["Repeat Customer", "Recent High-Value", "Lapsed High-Value", "Other One-Time"]
SEGMENT_COLORS = ["#5BA37C", "#2E7D9A", "#E08E2B", "#B0B0B0"]
ZONES = ["Zona Aman (<= rata-rata)", "Zona Waspada (rata-rata - 12%)", "Zona Kritis (> 12%)"]
ZONE_COLORS = dict(zip(ZONES, ["#5BA37C", "#E6B422", "#D9534F"]))
CRITICAL_LATE_PCT = 12  # batas Zona Kritis (%)
MIN_ORDERS_STATE = 500  # state dengan < 500 pesanan = sampel kecil
VOLUME_LABELS = ["Kecil (<500)", "Menengah (500-3.000)", "Besar (>3.000)"]
# Skala warna peta sama dengan notebook (kuning muda -> oranye -> merah tua)
LATE_CMAP = LinearSegmentedColormap.from_list("late", ["#FFF3B0", "#FD8D3C", "#BD0026"])


def show_fig(fig):
    """Tampilkan figure lalu tutup agar memori tidak menumpuk pada setiap rerun."""
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Memuat data...")
def load_data() -> pd.DataFrame:
    """1 baris = 1 item terjual (sesuai main_data.csv)."""
    df = pd.read_csv(DATA_PATH, usecols=USE_COLS, parse_dates=["order_purchase_timestamp"])
    df["order_date"] = df["order_purchase_timestamp"].dt.normalize()
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    # Didefinisikan ulang dari delay_days agar tidak bergantung pada tipe data kolom di CSV
    df["is_late"] = df["delay_days"] > 0
    return df


@st.cache_data(show_spinner="Menyiapkan data pesanan...")
def load_orders() -> pd.DataFrame:
    """1 baris = 1 pesanan. Dihitung sekali saja, filter diterapkan setelahnya."""
    df = load_data()
    orders = (df.groupby("order_id", as_index=False)
              .agg(order_purchase_timestamp=("order_purchase_timestamp", "first"),
                   order_date=("order_date", "first"),
                   order_month=("order_month", "first"),
                   customer_unique_id=("customer_unique_id", "first"),
                   customer_state=("customer_state", "first"),
                   revenue=("price", "sum"),
                   review_score=("review_score", "first"),
                   delivery_days=("delivery_days", "first"),
                   delay_days=("delay_days", "first"),
                   is_late=("is_late", "first")))
    orders["delay_group"] = pd.cut(orders["delay_days"], bins=[-np.inf, 0, 3, 7, np.inf], labels=DELAY_LABELS)
    return orders


@st.cache_data(show_spinner="Menghitung RFM...")
def make_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    snapshot = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (orders.groupby("customer_unique_id")
           .agg(last_purchase=("order_purchase_timestamp", "max"),
                frequency=("order_id", "nunique"),
                monetary=("revenue", "sum"))
           .reset_index())
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Vektorisasi (jauh lebih cepat daripada apply per baris pada ~93 ribu pelanggan)
    rfm["segment"] = np.select(
        [rfm["frequency"] >= 2,
         (rfm["r_score"] >= 4) & (rfm["m_score"] >= 4),
         (rfm["r_score"] <= 2) & (rfm["m_score"] >= 4)],
        SEGMENTS[:3],
        default=SEGMENTS[3],
    )
    return rfm


# --------------------------------------------------------------------------- sidebar
data = load_data()
all_orders = load_orders()

st.sidebar.title("Filter")
min_date, max_date = data["order_date"].min().date(), data["order_date"].max().date()
date_range = st.sidebar.date_input("Periode pembelian", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
all_states = sorted(data["customer_state"].dropna().unique())
selected_states = st.sidebar.multiselect("State pelanggan", options=all_states, default=all_states)

st.title("Dashboard Analisis E-Commerce (Olist)")
st.caption("Pesanan berstatus *delivered*, Januari 2017 - Agustus 2018. Revenue dihitung dari harga item (tanpa ongkos kirim) dalam R$.")

if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir pada filter periode.")
    st.stop()

start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
# Item dan pesanan membawa tanggal serta state yang sama, jadi filternya identik
df = data[data["order_date"].between(start_date, end_date) & data["customer_state"].isin(selected_states)]
orders = all_orders[all_orders["order_date"].between(start_date, end_date)
                    & all_orders["customer_state"].isin(selected_states)]
if df.empty or orders.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# Ringkasan per state dan rata-rata keterlambatan dipakai di tab Pengiriman dan Peta
overall_late = orders["is_late"].mean() * 100
state = (orders.groupby("customer_state")
         .agg(total_orders=("order_id", "count"), late_orders=("is_late", "sum"),
              avg_delivery_days=("delivery_days", "mean"),
              avg_review_score=("review_score", "mean")))
state["late_pct"] = state["late_orders"] / state["total_orders"] * 100

# --------------------------------------------------------------------------- KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"R$ {df['price'].sum() / 1e6:,.2f} juta")
k2.metric("Total Pesanan", f"{orders['order_id'].nunique():,}")
k3.metric("Rata-rata Review", f"{orders['review_score'].mean():.2f} / 5")
k4.metric("Pesanan Terlambat", f"{overall_late:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Kategori Produk", "Tren Bulanan", "Pengiriman", "Peta Wilayah", "Pelanggan (RFM)"])

# --------------------------------------------------------------------------- Pertanyaan 1
with tab1:
    st.subheader("Kategori produk dengan revenue tertinggi dan review score-nya")
    top_n = st.slider("Jumlah kategori ditampilkan", 5, 15, 10)

    cat = (df.groupby("product_category_name_english")
           .agg(total_revenue=("price", "sum"), total_orders=("order_id", "nunique"),
                units_sold=("order_item_id", "count"), avg_price=("price", "mean"),
                avg_review_score=("review_score", "mean"))
           .sort_values("total_revenue", ascending=False))
    cat["revenue_share_pct"] = cat["total_revenue"] / cat["total_revenue"].sum() * 100
    cat["cumulative_share_pct"] = cat["revenue_share_pct"].cumsum()
    top = cat.head(top_n).iloc[::-1]
    labels = top.index.str.replace("_", " ").str.title()
    overall_review = df["review_score"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(top) + 2), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    bars = axes[0].barh(labels, top["total_revenue"] / 1e6, color="#2E7D9A")
    for bar, share in zip(bars, top["revenue_share_pct"]):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.2f} ({share:.1f}%)", va="center", fontsize=9)
    axes[0].set_xlim(0, top["total_revenue"].max() / 1e6 * 1.22)
    axes[0].set_title("Total Revenue (R$ juta)", fontweight="bold")

    colors = ["#D9534F" if v < overall_review else "#5BA37C" for v in top["avg_review_score"]]
    bars2 = axes[1].barh(labels, top["avg_review_score"], color=colors)
    axes[1].axvline(overall_review, color="black", linestyle="--", linewidth=1.2, label=f"Rata-rata: {overall_review:.2f}")
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
    axes[1].set_xlim(0, 5)
    axes[1].set_title("Rata-rata Review Score", fontweight="bold")
    for bar in bars2:
        axes[1].text(bar.get_width() - 0.08, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}",
                     va="center", ha="right", color="white", fontweight="bold", fontsize=9)
    plt.tight_layout()
    show_fig(fig)
    st.caption(f"{top_n} kategori teratas menyumbang {cat['revenue_share_pct'].head(top_n).sum():.1f}% total revenue. "
               "Hijau = review setara/di atas rata-rata seluruh item; merah = di bawah rata-rata. "
               "Urutan revenue sensitif terhadap item berharga tinggi (outlier harga).")
    with st.expander("Lihat tabel lengkap seluruh kategori"):
        st.dataframe(cat.round(2))

# --------------------------------------------------------------------------- Pertanyaan 2
with tab2:
    st.subheader("Tren pesanan dan revenue bulanan")
    monthly = (orders.groupby("order_month")
               .agg(total_orders=("order_id", "nunique"), total_revenue=("revenue", "sum"),
                    avg_order_value=("revenue", "mean")))
    monthly["orders_mom_pct"] = monthly["total_orders"].pct_change() * 100
    monthly["revenue_mom_pct"] = monthly["total_revenue"].pct_change() * 100
    monthly["orders_mom_abs"] = monthly["total_orders"].diff()
    monthly["revenue_mom_abs"] = monthly["total_revenue"].diff()
    x_labels = monthly.index.astype(str)
    months = list(monthly.index)
    has_mom = len(monthly) > 1

    peak = monthly["total_orders"].idxmax()
    peak_rev = monthly["total_revenue"].idxmax()
    peak_pos = months.index(peak)
    peak_rev_pos = months.index(peak_rev)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pesanan tertinggi", str(peak), f"{monthly.loc[peak, 'total_orders']:,} pesanan", delta_color="off")
    c2.metric("Revenue tertinggi", str(peak_rev), f"R$ {monthly.loc[peak_rev, 'total_revenue']:,.0f}", delta_color="off")
    if has_mom:
        mom_peak = monthly["orders_mom_pct"].idxmax()
        mom_peak_pos = months.index(mom_peak)
        c3.metric("Kenaikan MoM tertinggi (%)", str(mom_peak), f"{monthly.loc[mom_peak, 'orders_mom_pct']:+.1f}% pesanan")
        c4.metric("Rata-rata pertumbuhan pesanan MoM", f"{monthly['orders_mom_pct'].mean():.1f}%")
    else:
        c3.metric("Kenaikan MoM tertinggi (%)", "-")
        c4.metric("Rata-rata pertumbuhan pesanan MoM", "-")

    # Pembanding tahunan pada periode yang sama (Jan-Agu); hanya bila kedua tahun ada pada filter
    jan_aug = orders[orders["order_purchase_timestamp"].dt.month <= 8]
    yoy = jan_aug.groupby(jan_aug["order_purchase_timestamp"].dt.year).agg(
        total_orders=("order_id", "nunique"), total_revenue=("revenue", "sum"))
    if {2017, 2018} <= set(yoy.index):
        g_ord = (yoy.loc[2018, "total_orders"] / yoy.loc[2017, "total_orders"] - 1) * 100
        g_rev = (yoy.loc[2018, "total_revenue"] / yoy.loc[2017, "total_revenue"] - 1) * 100
        y1, y2 = st.columns(2)
        y1.metric("Pesanan Jan-Agu 2018 vs Jan-Agu 2017", f"{yoy.loc[2018, 'total_orders']:,}", f"{g_ord:+.2f}%")
        y2.metric("Revenue Jan-Agu 2018 vs Jan-Agu 2017", f"R$ {yoy.loc[2018, 'total_revenue'] / 1e6:,.2f} juta", f"{g_rev:+.2f}%")
    else:
        st.caption("Perbandingan Jan-Agu 2018 vs 2017 muncul bila filter periode mencakup kedua tahun.")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1]})
    # (a) Jumlah pesanan
    axes[0].plot(x_labels, monthly["total_orders"], marker="o", color="#2E7D9A", linewidth=2.2)
    axes[0].set_title("Jumlah Pesanan per Bulan", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Jumlah pesanan")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].annotate(f"Puncak: {monthly.loc[peak, 'total_orders']:,} pesanan ({peak})",
                     xy=(peak_pos, monthly.loc[peak, "total_orders"]),
                     xytext=(max(0, peak_pos - 5.5), monthly.loc[peak, "total_orders"] * 0.93),
                     arrowprops=dict(arrowstyle="->", color="#D9534F"), color="#D9534F", fontsize=11, fontweight="bold")
    # (b) Revenue
    axes[1].plot(x_labels, monthly["total_revenue"] / 1e6, marker="o", color="#E08E2B", linewidth=2.2)
    axes[1].set_title("Total Revenue per Bulan", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Revenue (juta BRL)")
    axes[1].annotate(f"Puncak: {monthly.loc[peak_rev, 'total_revenue'] / 1e6:.2f} juta BRL ({peak_rev})",
                     xy=(peak_rev_pos, monthly.loc[peak_rev, "total_revenue"] / 1e6),
                     xytext=(max(0, peak_rev_pos - 5.5), monthly.loc[peak_rev, "total_revenue"] / 1e6 * 0.93),
                     arrowprops=dict(arrowstyle="->", color="#D9534F"), color="#D9534F", fontsize=11, fontweight="bold")
    # (c) Pertumbuhan MoM pesanan (%)
    mom = monthly["orders_mom_pct"].fillna(0)
    axes[2].bar(x_labels, mom, color=["#5BA37C" if v >= 0 else "#D9534F" for v in mom])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Pertumbuhan Jumlah Pesanan Month-over-Month (%)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("MoM (%)")
    axes[2].set_xlabel("Bulan")
    if has_mom:
        prev = months[mom_peak_pos - 1]
        axes[2].annotate(f"Tertinggi (%): {monthly.loc[mom_peak, 'orders_mom_pct']:+.1f}% ({mom_peak})\n"
                         f"(basis kecil: {monthly.loc[prev, 'total_orders']:,} -> {monthly.loc[mom_peak, 'total_orders']:,} pesanan)",
                         xy=(mom_peak_pos, monthly.loc[mom_peak, "orders_mom_pct"]),
                         xytext=(mom_peak_pos + 2, monthly.loc[mom_peak, "orders_mom_pct"] * 0.85),
                         arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=10, fontweight="bold")
        abs_peak = monthly["orders_mom_abs"].idxmax()
        abs_pos = months.index(abs_peak)
        if abs_peak != mom_peak:
            axes[2].annotate(f"{monthly.loc[abs_peak, 'orders_mom_pct']:+.1f}% ({abs_peak})\n"
                             f"kenaikan absolut terbesar: +{monthly.loc[abs_peak, 'orders_mom_abs']:,.0f} pesanan",
                             xy=(abs_pos, monthly.loc[abs_peak, "orders_mom_pct"]),
                             xytext=(max(0, abs_pos - 5.5), monthly.loc[abs_peak, "orders_mom_pct"] * 1.05),
                             arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=10, fontweight="bold")
        axes[2].margins(y=0.25)
    plt.setp(axes[2].get_xticklabels(), rotation=45)
    fig.suptitle(f"Tren Pesanan dan Revenue Bulanan ({x_labels[0]} - {x_labels[-1]})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    show_fig(fig)
    st.caption("Kenaikan MoM tertinggi secara persentase sering terjadi karena basis yang kecil; kenaikan absolut terbesar "
               "menunjukkan lonjakan volume sebenarnya. Nilai rata-rata per pesanan relatif stabil, sehingga pertumbuhan "
               "berasal dari jumlah pesanan. Penyebab lonjakan dan pola musiman tidak dapat dipastikan (data hanya memuat satu "
               "bulan November dan tidak ada data kampanye). Bulan pertama/terakhir pada filter mungkin tidak penuh.")
    with st.expander("Lihat tabel bulanan"):
        st.dataframe(monthly.round(2))

# --------------------------------------------------------------------------- Pertanyaan 3
with tab3:
    st.subheader("Dampak keterlambatan terhadap review dan state dengan keterlambatan tertinggi")
    late = orders[orders["is_late"]]["review_score"].mean()
    ontime = orders[~orders["is_late"]]["review_score"].mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Review pesanan tepat waktu", f"{ontime:.2f}" if pd.notna(ontime) else "-")
    m2.metric("Review pesanan terlambat", f"{late:.2f}" if pd.notna(late) else "-")
    m3.metric("Selisih", f"{ontime - late:.2f} poin" if pd.notna(late) and pd.notna(ontime) else "-")

    delay = (orders.groupby("delay_group", observed=True)
             .agg(total_orders=("order_id", "count"), avg_review_score=("review_score", "mean"),
                  pct_review_1=("review_score", lambda s: (s == 1).mean() * 100)))
    min_orders = st.number_input("Minimal jumlah pesanan per state (agar persentase tidak bias)", 0, 5000, MIN_ORDERS_STATE, step=100)
    top_late = state[state["total_orders"] >= min_orders].sort_values("late_pct", ascending=False).head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_colors = [DELAY_COLORS[str(g)] for g in delay.index]  # warna mengikuti label, bukan urutan
    bars = axes[0].bar(delay.index.astype(str), delay["avg_review_score"], color=bar_colors)
    axes[0].set_ylim(0, 5)
    axes[0].set_title("Rata-rata Review Score menurut Ketepatan Pengiriman", fontweight="bold")
    for bar, n, p1 in zip(bars, delay["total_orders"], delay["pct_review_1"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.2f}",
                     ha="center", fontsize=9)
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"n={n:,}\nSkor 1: {p1:.0f}%",
                     ha="center", va="center", fontsize=8.5)
    plt.setp(axes[0].get_xticklabels(), rotation=15)

    if top_late.empty:
        axes[1].text(0.5, 0.5, "Tidak ada state yang memenuhi\nminimal jumlah pesanan", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        bars2 = axes[1].barh(top_late.index, top_late["late_pct"], color="#D9534F")
        axes[1].axvline(overall_late, color="black", linestyle="--", linewidth=1.2, label=f"Rata-rata: {overall_late:.1f}%")
        axes[1].legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
        for bar, rv in zip(bars2, top_late["avg_review_score"]):
            axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         f"{bar.get_width():.1f}%  (review {rv:.2f})", va="center", fontsize=9)
        axes[1].set_xlim(0, max(top_late["late_pct"].max() * 1.4, 1))
        axes[1].set_title(f"State dengan Persentase Keterlambatan Tertinggi\n(state dengan >= {min_orders:,} pesanan)", fontweight="bold")
        axes[1].set_xlabel("Pesanan terlambat (%)")
    plt.tight_layout()
    show_fig(fig)
    all_top = state.sort_values("late_pct", ascending=False).iloc[0]
    st.caption(f"Dari seluruh state, keterlambatan tertinggi ada di {all_top.name} ({all_top['late_pct']:.2f}%, "
               f"{int(all_top['total_orders']):,} pesanan). Grafik hanya memuat state dengan >= {min_orders:,} pesanan agar "
               "persentase tidak bias sampel kecil. Hubungan keterlambatan dan review adalah korelasi, bukan sebab-akibat.")

# --------------------------------------------------------------------------- Geospasial + zona
with tab4:
    st.subheader("Peta pesanan dan keterlambatan per state")
    geo = state.reset_index()
    geo["lat"] = geo["customer_state"].map(lambda s: STATE_COORDS[s][0])
    geo["lon"] = geo["customer_state"].map(lambda s: STATE_COORDS[s][1])
    norm_late = (geo["late_pct"] - geo["late_pct"].min()) / max(geo["late_pct"].max() - geo["late_pct"].min(), 1e-9)
    geo["color"] = ["#%02x%02x%02x" % tuple(int(c * 255) for c in LATE_CMAP(v)[:3]) for v in norm_late]
    geo["size"] = np.maximum(np.sqrt(geo["total_orders"]) * 1200, 25000)  # skala akar kuadrat agar SP tidak menutupi state lain

    st.map(geo, latitude="lat", longitude="lon", size="size", color="color")
    st.caption("Ukuran lingkaran = jumlah pesanan (skala akar kuadrat). Warna: kuning = keterlambatan rendah, "
               "merah tua = tinggi. Koordinat adalah titik pusat rata-rata tiap state, bukan lokasi pelanggan.")

    # Binning zona logistik. np.select dipakai (bukan pd.cut) agar tidak error bila rata-rata hasil filter > 12%.
    geo["zona_logistik"] = np.select(
        [geo["late_pct"] <= overall_late, geo["late_pct"] <= CRITICAL_LATE_PCT],
        ZONES[:2], default=ZONES[2])
    geo["kategori_volume"] = pd.cut(geo["total_orders"], bins=[0, 500, 3000, np.inf], labels=VOLUME_LABELS)

    st.markdown(f"**Zona logistik** (Aman <= rata-rata {overall_late:.1f}%; "
                f"Waspada sampai {CRITICAL_LATE_PCT}%; Kritis > {CRITICAL_LATE_PCT}%)")
    zona = (geo.groupby("zona_logistik")
            .agg(jumlah_state=("customer_state", "count"), total_pesanan=("total_orders", "sum"),
                 pesanan_terlambat=("late_orders", "sum"), avg_review_score=("avg_review_score", "mean"),
                 avg_delivery_days=("avg_delivery_days", "mean"),
                 daftar_state=("customer_state", lambda s: ", ".join(sorted(s)))))
    zona["late_pct_zona"] = zona["pesanan_terlambat"] / zona["total_pesanan"] * 100
    zona["share_pesanan_pct"] = zona["total_pesanan"] / zona["total_pesanan"].sum() * 100
    st.dataframe(zona.reindex([z for z in ZONES if z in zona.index]).round(2))
    st.caption("Rata-rata review dan lama pengiriman pada tabel zona adalah rata-rata antar-state (tidak dibobot jumlah pesanan).")

    plot_df = geo.sort_values("late_pct")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(plot_df) + 1.5)))
    ax.barh(plot_df["customer_state"], plot_df["late_pct"], color=plot_df["zona_logistik"].map(ZONE_COLORS))
    line1 = ax.axvline(overall_late, color="black", linestyle="--", linewidth=1.2)
    line2 = ax.axvline(CRITICAL_LATE_PCT, color="darkblue", linestyle=":", linewidth=1.8)
    for i, (p, n) in enumerate(zip(plot_df["late_pct"], plot_df["total_orders"])):
        ax.text(p + 0.3, i, f"{p:.1f}%  (n={n:,})", va="center", fontsize=9)
    ax.set_xlim(0, max(plot_df["late_pct"].max() * 1.25, CRITICAL_LATE_PCT * 1.1))
    ax.set_xlabel("Pesanan terlambat (%)")
    ax.set_title("Klasterisasi State ke dalam Zona Logistik berdasarkan Keterlambatan", fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ZONE_COLORS.values()] + [line1, line2]
    ax.legend(handles, list(ZONE_COLORS) + [f"Rata-rata ({overall_late:.1f}%)", f"Batas Kritis ({CRITICAL_LATE_PCT}%)"],
              loc="lower right", frameon=True, facecolor="white", fontsize=9)
    plt.tight_layout()
    show_fig(fig)

    st.markdown(f"**State prioritas** (Zona Kritis dengan >= {MIN_ORDERS_STATE} pesanan)")
    prio = geo[(geo["zona_logistik"] == ZONES[2]) & (geo["total_orders"] >= MIN_ORDERS_STATE)]
    if prio.empty:
        st.info("Tidak ada state Zona Kritis dengan volume cukup pada filter ini.")
    else:
        st.dataframe(prio.sort_values("total_orders", ascending=False)
                     [["customer_state", "total_orders", "late_pct", "avg_review_score", "kategori_volume"]].round(2),
                     hide_index=True)
    st.caption("State Zona Kritis dengan < 500 pesanan sebaiknya dipantau dulu karena persentasenya mudah berubah.")
    with st.expander("Detail seluruh state"):
        st.dataframe(geo[["customer_state", "total_orders", "late_pct", "avg_delivery_days", "avg_review_score",
                          "zona_logistik", "kategori_volume"]].sort_values("late_pct", ascending=False).round(2))

# --------------------------------------------------------------------------- RFM
with tab5:
    st.subheader("Segmentasi pelanggan (RFM)")
    if orders["customer_unique_id"].nunique() < 5:
        st.info("Data terlalu sedikit untuk analisis RFM. Perluas filter periode atau state.")
    else:
        rfm = make_rfm(orders)
        r1, r2, r3 = st.columns(3)
        r1.metric("Pelanggan unik", f"{len(rfm):,}")
        r2.metric("Pelanggan repeat order", f"{(rfm['frequency'] >= 2).mean() * 100:.1f}%")
        r3.metric("Rata-rata recency", f"{rfm['recency'].mean():.0f} hari")

        seg = (rfm.groupby("segment")
               .agg(customers=("customer_unique_id", "count"), avg_recency_days=("recency", "mean"),
                    avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"), total_monetary=("monetary", "sum"))
               .reindex(SEGMENTS).fillna(0))
        seg["customer_share_pct"] = seg["customers"] / seg["customers"].sum() * 100
        seg["revenue_share_pct"] = seg["total_monetary"] / seg["total_monetary"].sum() * 100

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, col, title in zip(axes, ["customer_share_pct", "revenue_share_pct"],
                                  ["Porsi Pelanggan per Segmen (%)", "Porsi Revenue per Segmen (%)"]):
            ax.bar(SEGMENTS, seg[col], color=SEGMENT_COLORS)
            ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, max(seg[col].max() * 1.15, 1))
            for i, v in enumerate(seg[col]):
                ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=15)
        plt.tight_layout()
        show_fig(fig)
        st.dataframe(seg.round(2))
        st.caption("Repeat Customer = >= 2 pesanan; Recent High-Value = skor recency & monetary 4-5; Lapsed High-Value = "
                   "recency 1-2 & monetary 4-5; sisanya Other One-Time. Skor 1-5 (5 = terbaik).")

st.caption("Sumber data: Olist Brazilian E-Commerce Public Dataset. Dashboard dibuat dengan Streamlit.")
