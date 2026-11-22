from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
ZONES = ["Zona Aman", "Zona Waspada", "Zona Kritis"]
CRITICAL_LATE_PCT = 12  # batas Zona Kritis (%)


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
st.caption("Pesanan berstatus *delivered*. Revenue dihitung dari harga item (tanpa ongkos kirim) dalam R$.")

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
           .agg(total_revenue=("price", "sum"), units_sold=("order_item_id", "count"),
                avg_review_score=("review_score", "mean"))
           .sort_values("total_revenue", ascending=False))
    cat["revenue_share_pct"] = cat["total_revenue"] / cat["total_revenue"].sum() * 100
    top = cat.head(top_n).iloc[::-1]
    labels = top.index.str.replace("_", " ").str.title()
    overall_review = df["review_score"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(top) + 2), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    bars = axes[0].barh(labels, top["total_revenue"] / 1e6, color="#2E7D9A")
    for bar, share in zip(bars, top["revenue_share_pct"]):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.2f} ({share:.1f}%)", va="center", fontsize=9)
    axes[0].set_xlim(0, top["total_revenue"].max() / 1e6 * 1.25)
    axes[0].set_title("Total Revenue (R$ juta)", fontweight="bold")

    colors = ["#D9534F" if v < overall_review else "#5BA37C" for v in top["avg_review_score"]]
    bars2 = axes[1].barh(labels, top["avg_review_score"], color=colors)
    axes[1].axvline(overall_review, color="black", linestyle="--", linewidth=1.2, label=f"Rata-rata: {overall_review:.2f}")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_xlim(0, 5)
    axes[1].set_title("Rata-rata Review Score", fontweight="bold")
    for bar in bars2:
        axes[1].text(bar.get_width() - 0.08, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}",
                     va="center", ha="right", color="white", fontweight="bold", fontsize=9)
    plt.tight_layout()
    show_fig(fig)
    st.caption("Hijau = review setara/di atas rata-rata seluruh item; merah = di bawah rata-rata.")
    with st.expander("Lihat tabel lengkap seluruh kategori"):
        st.dataframe(cat.round(2))

# --------------------------------------------------------------------------- Pertanyaan 2
with tab2:
    st.subheader("Tren pesanan dan revenue bulanan")
    monthly = (orders.groupby("order_month")
               .agg(total_orders=("order_id", "nunique"), total_revenue=("revenue", "sum")))
    monthly["orders_mom_pct"] = monthly["total_orders"].pct_change() * 100
    x_labels = monthly.index.astype(str)
    peak = monthly["total_orders"].idxmax()
    peak_pos = list(monthly.index).index(peak)
    mom_val = monthly.loc[peak, "orders_mom_pct"]
    mom_text = f", MoM {mom_val:+.0f}%" if pd.notna(mom_val) else ""
    c1, c2 = st.columns(2)
    c1.metric("Bulan dengan pesanan tertinggi", str(peak), f"{monthly.loc[peak, 'total_orders']:,} pesanan")
    c2.metric("Rata-rata pertumbuhan pesanan MoM", f"{monthly['orders_mom_pct'].mean():.1f}%" if len(monthly) > 1 else "-")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    # Subplot 1: Jumlah Pesanan per Bulan
    axes[0].plot(x_labels, monthly["total_orders"], marker="o", color="#2E7D9A", linewidth=2.2)
    axes[0].set_title("Jumlah Pesanan per Bulan", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Jumlah pesanan")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].annotate(
        f"Puncak: {monthly.loc[peak, 'total_orders']:,} pesanan\n({peak}{mom_text})",
        xy=(peak_pos, monthly.loc[peak, "total_orders"]),
        xytext=(max(0, peak_pos - 4.2), monthly.loc[peak, "total_orders"] * 0.93),
        arrowprops=dict(arrowstyle="->", color="#D9534F"),
        color="#D9534F", fontsize=11, fontweight="bold"
    )
    # Subplot 2: Total Revenue per Bulan
    axes[1].plot(x_labels, monthly["total_revenue"] / 1e6, marker="o", color="#E08E2B", linewidth=2.2)
    axes[1].set_title("Total Revenue per Bulan", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Revenue (R$ juta)")
    axes[1].set_xlabel("Bulan")
    axes[1].annotate(
        f"Puncak: R$ {monthly.loc[peak, 'total_revenue'] / 1e6:.2f} juta",
        xy=(peak_pos, monthly.loc[peak, "total_revenue"] / 1e6),
        xytext=(max(0, peak_pos - 4.2), monthly.loc[peak, "total_revenue"] / 1e6 * 0.93),
        arrowprops=dict(arrowstyle="->", color="#D9534F"),
        color="#D9534F", fontsize=11, fontweight="bold"
    )
    plt.setp(axes[1].get_xticklabels(), rotation=45)
    fig.suptitle(f"Tren Pesanan dan Revenue Bulanan ({x_labels[0]} - {x_labels[-1]})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    show_fig(fig)
    st.caption("Bulan pertama/terakhir pada rentang filter mungkin tidak penuh satu bulan.")
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
    min_orders = st.number_input("Minimal jumlah pesanan per state (agar persentase tidak bias)", 0, 5000, 500, step=100)
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
        axes[1].legend(loc="lower right", fontsize=9)
        for bar, rv in zip(bars2, top_late["avg_review_score"]):
            axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                         f"{bar.get_width():.1f}%  (review {rv:.2f})", va="center", fontsize=9)
        axes[1].set_xlim(0, max(top_late["late_pct"].max() * 1.4, 1))
        axes[1].set_title("State dengan Persentase Keterlambatan Tertinggi", fontweight="bold")
        axes[1].set_xlabel("Pesanan terlambat (%)")
    plt.tight_layout()
    show_fig(fig)

# --------------------------------------------------------------------------- Geospasial + zona
with tab4:
    st.subheader("Peta pesanan dan keterlambatan per state")
    geo = state.reset_index()
    geo["lat"] = geo["customer_state"].map(lambda s: STATE_COORDS[s][0])
    geo["lon"] = geo["customer_state"].map(lambda s: STATE_COORDS[s][1])
    cmap = plt.get_cmap("YlOrRd")
    norm_late = (geo["late_pct"] - geo["late_pct"].min()) / max(geo["late_pct"].max() - geo["late_pct"].min(), 1e-9)
    geo["color"] = ["#%02x%02x%02x" % tuple(int(c * 255) for c in cmap(v)[:3]) for v in norm_late]
    geo["size"] = np.maximum(np.sqrt(geo["total_orders"]) * 1200, 25000)

    st.map(geo, latitude="lat", longitude="lon", size="size", color="color")
    st.caption("Ukuran lingkaran = jumlah pesanan. Warna: kuning = keterlambatan rendah, merah tua = tinggi.")

    # Clustering manual: zona logistik (binning). np.select dipakai, bukan pd.cut, karena pd.cut
    # error bila rata-rata keterlambatan hasil filter lebih besar dari batas 12%.
    geo["zona_logistik"] = np.select(
        [geo["late_pct"] <= overall_late, geo["late_pct"] <= CRITICAL_LATE_PCT],
        ["Zona Aman", "Zona Waspada"],
        default="Zona Kritis",
    )
    st.markdown(f"**Zona logistik** (Aman <= rata-rata {overall_late:.1f}%; "
                f"Waspada sampai {CRITICAL_LATE_PCT}%; Kritis > {CRITICAL_LATE_PCT}%)")
    zona = (geo.groupby("zona_logistik")
            .agg(jumlah_state=("customer_state", "count"), total_pesanan=("total_orders", "sum"),
                 avg_review_score=("avg_review_score", "mean"),
                 daftar_state=("customer_state", lambda s: ", ".join(sorted(s)))))
    st.dataframe(zona.reindex([z for z in ZONES if z in zona.index]).round(2))
    with st.expander("Detail seluruh state"):
        st.dataframe(geo[["customer_state", "total_orders", "late_pct", "avg_review_score", "zona_logistik"]]
                     .sort_values("late_pct", ascending=False).round(2))

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
                    avg_monetary=("monetary", "mean"), total_monetary=("monetary", "sum"))
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

st.caption("Sumber data: Olist Brazilian E-Commerce Public Dataset. Dashboard dibuat dengan Streamlit.")