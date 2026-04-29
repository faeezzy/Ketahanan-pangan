import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="PanganWatch — Ketahanan Pangan Indonesia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0f1117;
    color: #e8e8e8;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a8e063, #56ab2f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #1e2535);
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #a8e063;
}

.metric-label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.cluster-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8e8e8;
    border-left: 3px solid #a8e063;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

[data-testid="stSidebar"] {
    background-color: #0d1018;
    border-right: 1px solid #1e2535;
}

.stSelectbox label, .stMultiSelect label {
    color: #aaa !important;
    font-size: 0.85rem !important;
}

div[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 10px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────
@st.cache_data
def load_harga():
    df = pd.read_csv('output_harga_beras_clean.csv')
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    df = df.dropna(subset=['tanggal', 'harga'])
    return df

@st.cache_data
def load_supply():
    df = pd.read_csv('output_supply_nasional.csv')
    return df

@st.cache_data
def load_clustering():
    df = pd.read_csv('output_clustering_provinsi.csv')
    return df

@st.cache_data
def build_model(df_harga, df_supply):
    df_medium = df_harga[df_harga['komoditas'] == 'Beras Kualitas Medium I'].copy()
    df_merged = pd.merge(df_medium, df_supply, on=['tahun', 'bulan'], how='inner')
    if len(df_merged) < 4:
        return None, None, None, None
    X = df_merged[['total_produksi_ton', 'total_luas_panen_ha', 'avg_yield']].values
    y = df_merged['harga'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, df_merged, y, y_pred

try:
    df_harga = load_harga()
    df_supply = load_supply()
    df_cluster = load_clustering()
    model, df_merged, y_actual, y_pred = build_model(df_harga, df_supply)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading data: {e}")

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem'>🌾</div>
        <div style='font-family:Syne,sans-serif; font-weight:800; font-size:1.2rem; color:#a8e063'>PanganWatch</div>
        <div style='font-size:0.75rem; color:#666; margin-top:0.2rem'>Dashboard Ketahanan Pangan</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Overview", "📈 Tren Harga Beras", "🗺️ Peta Ketahanan Provinsi", "🤖 Prediksi Harga", "📊 Supply Nasional"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#555; line-height:1.6'>
    <b style='color:#777'>Sumber Data</b><br>
    • BPS — Produksi & Luas Panen Padi 2025<br>
    • Kemendag — Harga Beras 2024–2026<br>
    • BPS — IHK Makanan 2024–2026
    </div>
    """, unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# ── CLUSTER COLORS ───────────────────────────────────────────
CLUSTER_COLORS = {
    'Lumbung Padi':       '#a8e063',
    'Produksi Tinggi':    '#56ab2f',
    'Produksi Menengah':  '#f0a500',
    'Produksi Rendah':    '#e05c5c',
}

BULAN_LABELS = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
                7:'Jul',8:'Ags',9:'Sep',10:'Okt',11:'Nov',12:'Des'}

# ════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-title">PanganWatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sistem Pemantauan & Analisis Ketahanan Pangan Indonesia</div>', unsafe_allow_html=True)

    # KPI Cards
    harga_terbaru = df_harga[df_harga['komoditas']=='Beras Kualitas Medium I'].sort_values('tanggal').iloc[-1]['harga']
    harga_awal    = df_harga[df_harga['komoditas']=='Beras Kualitas Medium I'].sort_values('tanggal').iloc[0]['harga']
    delta_harga   = ((harga_terbaru - harga_awal) / harga_awal) * 100

    total_produksi = df_supply['total_produksi_ton'].sum() / 1e6
    avg_yield      = df_cluster['avg_yield'].mean()
    n_lumbung      = len(df_cluster[df_cluster['cluster_label'] == 'Lumbung Padi'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Harga Beras Medium I", f"Rp {harga_terbaru:,.0f}", f"{delta_harga:+.1f}% sejak 2024")
    with col2:
        st.metric("Total Produksi 2025", f"{total_produksi:.1f} Juta Ton", "38 provinsi")
    with col3:
        st.metric("Rata-rata Yield", f"{avg_yield:.2f} ton/ha", "Nasional 2025")
    with col4:
        st.metric("Provinsi Lumbung Padi", f"{n_lumbung} Provinsi", "Cluster tertinggi")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<div class="section-header">Tren Harga Beras (2024–2026)</div>', unsafe_allow_html=True)
        df_plot = df_harga[df_harga['komoditas'].isin(['Beras Kualitas Medium I','Beras Kualitas Super I','Beras Kualitas Bawah I'])].sort_values('tanggal')
        fig = px.line(df_plot, x='tanggal', y='harga', color='komoditas',
                      color_discrete_sequence=['#a8e063','#f0a500','#56d9f5'],
                      template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=11)),
            xaxis=dict(title='', gridcolor='#1e2535'),
            yaxis=dict(title='Rp/kg', gridcolor='#1e2535'),
            margin=dict(l=0, r=0, t=30, b=0), height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Distribusi Cluster Provinsi</div>', unsafe_allow_html=True)
        cluster_counts = df_cluster['cluster_label'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'jumlah']
        fig2 = px.pie(cluster_counts, values='jumlah', names='cluster',
                      color='cluster',
                      color_discrete_map=CLUSTER_COLORS,
                      hole=0.55, template='plotly_dark')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(size=11)),
            margin=dict(l=0, r=0, t=10, b=0), height=300
        )
        fig2.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig2, use_container_width=True)

    # Problem Statement
    st.markdown("---")
    st.markdown('<div class="section-header">Tentang Dashboard Ini</div>', unsafe_allow_html=True)
    st.markdown("""
    Dashboard ini menganalisis **stabilitas rantai pasok pangan Indonesia** dari sisi supply (produksi & luas panen)
    hingga dampaknya ke harga konsumen (IHK Makanan). Pendekatan yang digunakan:

    - **Clustering K-Means** untuk mengidentifikasi profil ketahanan pangan per provinsi
    - **Random Forest Regression** untuk memodelkan hubungan supply padi → harga beras
    - **Analisis tren** harga beras 6 kualitas selama 2024–2026

    > *"Dengan memahami pola ini, pemerintah dapat memprioritaskan intervensi supply-chain
    di provinsi yang paling rentan terhadap ketidakstabilan harga pangan."*
    """)

# ════════════════════════════════════════════════════════════
# PAGE 2: TREN HARGA
# ════════════════════════════════════════════════════════════
elif page == "📈 Tren Harga Beras":
    st.markdown('<div class="section-header">Tren Harga Beras per Kualitas (2024–2026)</div>', unsafe_allow_html=True)

    kualitas_options = df_harga['komoditas'].unique().tolist()
    selected_kualitas = st.multiselect(
        "Pilih kualitas beras:",
        kualitas_options,
        default=kualitas_options[:3]
    )

    if selected_kualitas:
        df_plot = df_harga[df_harga['komoditas'].isin(selected_kualitas)].sort_values('tanggal')
        fig = px.line(df_plot, x='tanggal', y='harga', color='komoditas',
                      markers=True, template='plotly_dark',
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Tanggal', gridcolor='#1e2535'),
            yaxis=dict(title='Harga (Rp/kg)', gridcolor='#1e2535',
                       tickformat=',.0f'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=420, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
        stats = df_harga[df_harga['komoditas'].isin(selected_kualitas)].groupby('komoditas')['harga'].agg(
            ['min','max','mean','std']
        ).round(0).reset_index()
        stats.columns = ['Komoditas','Harga Min (Rp)','Harga Max (Rp)','Rata-rata (Rp)','Std Dev']
        st.dataframe(stats, use_container_width=True, hide_index=True)

        # Volatilitas
        st.markdown('<div class="section-header">Volatilitas Harga (Std Dev)</div>', unsafe_allow_html=True)
        vol = df_harga[df_harga['komoditas'].isin(selected_kualitas)].groupby('komoditas')['harga'].std().reset_index()
        vol.columns = ['komoditas','volatilitas']
        vol = vol.sort_values('volatilitas', ascending=True)
        fig_vol = px.bar(vol, x='volatilitas', y='komoditas', orientation='h',
                         color='volatilitas', color_continuous_scale='RdYlGn_r',
                         template='plotly_dark')
        fig_vol.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Std Dev Harga (Rp)', gridcolor='#1e2535'),
            yaxis=dict(title=''), height=280,
            coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3: PETA KETAHANAN PROVINSI
# ════════════════════════════════════════════════════════════
elif page == "🗺️ Peta Ketahanan Provinsi":
    st.markdown('<div class="section-header">Profil Ketahanan Pangan per Provinsi</div>', unsafe_allow_html=True)

    col_filter, _ = st.columns([2, 3])
    with col_filter:
        cluster_filter = st.multiselect(
            "Filter cluster:",
            df_cluster['cluster_label'].unique().tolist(),
            default=df_cluster['cluster_label'].unique().tolist()
        )

    df_filtered = df_cluster[df_cluster['cluster_label'].isin(cluster_filter)]

    # Scatter plot produksi vs yield
    fig = px.scatter(
        df_filtered,
        x='total_produksi', y='avg_yield',
        color='cluster_label',
        size='total_luas',
        hover_name='provinsi',
        hover_data={
            'total_produksi': ':,.0f',
            'avg_yield': ':.3f',
            'std_yield': ':.3f',
            'total_luas': ':,.0f',
            'cluster_label': True
        },
        color_discrete_map=CLUSTER_COLORS,
        template='plotly_dark',
        labels={
            'total_produksi': 'Total Produksi (Ton)',
            'avg_yield': 'Rata-rata Yield (Ton/Ha)',
            'cluster_label': 'Cluster'
        }
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#1e2535', tickformat=',.0f'),
        yaxis=dict(gridcolor='#1e2535'),
        height=450, margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary cards
    st.markdown('<div class="section-header">Ringkasan per Cluster</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (label, color) in enumerate(CLUSTER_COLORS.items()):
        df_c = df_cluster[df_cluster['cluster_label'] == label]
        if len(df_c) == 0:
            continue
        with cols[i % 4]:
            st.markdown(f"""
            <div style='background:#1a1f2e; border:1px solid {color}40;
                        border-top: 3px solid {color}; border-radius:10px;
                        padding:1rem; margin-bottom:1rem'>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:{color}; font-size:0.85rem; margin-bottom:0.8rem'>
                    {label}
                </div>
                <div style='font-size:1.6rem; font-weight:700; color:#e8e8e8;
                            font-family:Syne,sans-serif'>{len(df_c)}</div>
                <div style='font-size:0.72rem; color:#666'>provinsi</div>
                <div style='margin-top:0.6rem; font-size:0.78rem; color:#aaa'>
                    Avg produksi: <b style='color:#e8e8e8'>{df_c['total_produksi'].mean()/1e6:.2f}M ton</b><br>
                    Avg yield: <b style='color:#e8e8e8'>{df_c['avg_yield'].mean():.2f} ton/ha</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tabel detail
    st.markdown('<div class="section-header">Detail Provinsi</div>', unsafe_allow_html=True)
    df_display = df_filtered[['provinsi','cluster_label','total_produksi','avg_yield','std_yield','total_luas','bulan_aktif']].copy()
    df_display.columns = ['Provinsi','Cluster','Total Produksi (Ton)','Avg Yield (Ton/Ha)','Volatilitas Yield','Total Luas (Ha)','Bulan Aktif']
    df_display['Total Produksi (Ton)'] = df_display['Total Produksi (Ton)'].map('{:,.0f}'.format)
    df_display['Total Luas (Ha)'] = df_display['Total Luas (Ha)'].map('{:,.0f}'.format)
    df_display['Avg Yield (Ton/Ha)'] = df_display['Avg Yield (Ton/Ha)'].map('{:.3f}'.format)
    df_display['Volatilitas Yield'] = df_display['Volatilitas Yield'].map('{:.3f}'.format)
    st.dataframe(df_display.sort_values('Cluster'), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 4: PREDIKSI HARGA
# ════════════════════════════════════════════════════════════
elif page == "🤖 Prediksi Harga":
    st.markdown('<div class="section-header">Model Prediksi Harga Beras (Random Forest)</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("Data overlap antara supply dan harga tidak cukup untuk melatih model.")
    else:
        r2 = r2_score(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}", "semakin mendekati 1 = lebih baik")
        col2.metric("MAE", f"Rp {mae:,.0f}/kg", "rata-rata error prediksi")
        col3.metric("Data Training", f"{len(y_actual)} bulan", "overlap 2025")

        # Actual vs Predicted
        st.markdown('<div class="section-header">Aktual vs Prediksi</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_merged['bulan'].map(BULAN_LABELS), y=y_actual,
            mode='lines+markers', name='Harga Aktual',
            line=dict(color='#a8e063', width=2.5),
            marker=dict(size=7)
        ))
        fig.add_trace(go.Scatter(
            x=df_merged['bulan'].map(BULAN_LABELS), y=y_pred,
            mode='lines+markers', name='Prediksi Model',
            line=dict(color='#f0a500', width=2.5, dash='dash'),
            marker=dict(size=7, symbol='diamond')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Bulan 2025', gridcolor='#1e2535'),
            yaxis=dict(title='Harga Rp/kg', gridcolor='#1e2535', tickformat=',.0f'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=350, margin=dict(l=0,r=0,t=30,b=0),
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        feat_names = ['Total Produksi', 'Luas Panen', 'Avg Yield']
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'Fitur': feat_names, 'Importance': importances}).sort_values('Importance')
        fig_imp = px.bar(df_imp, x='Importance', y='Fitur', orientation='h',
                         color='Importance', color_continuous_scale=['#1e2535','#a8e063'],
                         template='plotly_dark')
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Importance Score', gridcolor='#1e2535'),
            yaxis=dict(title=''),
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Simulasi prediksi
        st.markdown('<div class="section-header">Simulasi Prediksi</div>', unsafe_allow_html=True)
        st.caption("Masukkan nilai supply untuk memprediksi harga beras Medium I")
        c1, c2, c3 = st.columns(3)
        with c1:
            sim_prod = st.number_input("Total Produksi (Ton)", min_value=1e6, max_value=20e6,
                                        value=float(df_supply['total_produksi_ton'].mean()),
                                        step=100000.0, format="%.0f")
        with c2:
            sim_luas = st.number_input("Luas Panen (Ha)", min_value=100000.0, max_value=5000000.0,
                                        value=float(df_supply['total_luas_panen_ha'].mean()),
                                        step=10000.0, format="%.0f")
        with c3:
            sim_yield = st.number_input("Avg Yield (Ton/Ha)", min_value=2.0, max_value=10.0,
                                         value=float(df_supply['avg_yield'].mean()),
                                         step=0.1, format="%.2f")

        if st.button("🔮 Prediksi Harga", use_container_width=True):
            pred = model.predict([[sim_prod, sim_luas, sim_yield]])[0]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a1f2e,#1e2535);
                        border:1px solid #a8e06360; border-radius:12px;
                        padding:1.5rem; text-align:center; margin-top:1rem'>
                <div style='font-size:0.85rem; color:#888; margin-bottom:0.3rem'>
                    Prediksi Harga Beras Medium I
                </div>
                <div style='font-family:Syne,sans-serif; font-size:2.5rem;
                            font-weight:800; color:#a8e063'>
                    Rp {pred:,.0f} <span style='font-size:1rem; color:#888'>/kg</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 5: SUPPLY NASIONAL
# ════════════════════════════════════════════════════════════
elif page == "📊 Supply Nasional":
    st.markdown('<div class="section-header">Pola Supply Padi Nasional 2025</div>', unsafe_allow_html=True)

    bulan_labels = df_supply['bulan'].map(BULAN_LABELS)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Total Produksi (Juta Ton)',
                                        'Total Luas Panen (Juta Ha)',
                                        'Rata-rata Yield (Ton/Ha)'])

    fig.add_trace(go.Bar(x=bulan_labels, y=df_supply['total_produksi_ton']/1e6,
                         marker_color='#a8e063', name='Produksi'), row=1, col=1)
    fig.add_trace(go.Bar(x=bulan_labels, y=df_supply['total_luas_panen_ha']/1e6,
                         marker_color='#56d9f5', name='Luas Panen'), row=1, col=2)
    fig.add_trace(go.Scatter(x=bulan_labels, y=df_supply['avg_yield'],
                              mode='lines+markers', line=dict(color='#f0a500', width=2.5),
                              marker=dict(size=8), name='Yield'), row=1, col=3)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, showlegend=False,
        margin=dict(l=0,r=0,t=40,b=0)
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor='#1e2535', row=1, col=i)
        fig.update_yaxes(gridcolor='#1e2535', row=1, col=i)

    st.plotly_chart(fig, use_container_width=True)

    # Insight box
    peak_bulan = df_supply.loc[df_supply['total_produksi_ton'].idxmax(), 'bulan']
    peak_label  = BULAN_LABELS[peak_bulan]
    peak_val    = df_supply['total_produksi_ton'].max() / 1e6

    st.markdown(f"""
    <div style='background:#1a1f2e; border-left:3px solid #a8e063;
                padding:1rem 1.5rem; border-radius:0 10px 10px 0; margin:1rem 0'>
        <div style='font-family:Syne,sans-serif; font-weight:700; color:#a8e063;
                    font-size:0.85rem; margin-bottom:0.4rem'>💡 Insight</div>
        <div style='color:#ccc; font-size:0.9rem'>
            Puncak produksi padi nasional 2025 terjadi pada bulan <b style='color:#e8e8e8'>{peak_label}</b>
            dengan total <b style='color:#a8e063'>{peak_val:.2f} juta ton</b>.
            Pola ini mencerminkan siklus musim tanam utama Indonesia yang umumnya
            dipanen pada akhir kuartal pertama dan ketiga.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top provinsi
    st.markdown('<div class="section-header">Top 10 Provinsi Produsen 2025</div>', unsafe_allow_html=True)
    top10 = df_cluster.nlargest(10, 'total_produksi')[['provinsi','total_produksi','avg_yield','cluster_label']]
    fig2 = px.bar(top10.sort_values('total_produksi'),
                  x='total_produksi', y='provinsi',
                  orientation='h', color='cluster_label',
                  color_discrete_map=CLUSTER_COLORS,
                  template='plotly_dark',
                  labels={'total_produksi':'Total Produksi (Ton)','provinsi':'','cluster_label':'Cluster'})
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#1e2535', tickformat=',.0f'),
        height=380, margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)
