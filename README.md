# 🌾 PanganWatch
### Sistem Pemantauan & Analisis Ketahanan Pangan Indonesia

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Blob%20Storage-0078D4?logo=microsoftazure&logoColor=white)

> *Dari data produksi padi hingga harga di pasar — PanganWatch menghubungkan keduanya untuk mengidentifikasi wilayah paling rentan terhadap instabilitas harga pangan.*

---

## 📌 Latar Belakang

Harga beras Indonesia naik **9.2%** dalam dua tahun terakhir, sementara produktivitas lahan padi antar provinsi sangat tidak merata. Selama ini, data produksi pertanian dan data harga konsumen tersimpan di silo terpisah dan tidak pernah dianalisis secara terintegrasi — membuat pemerintah dan distributor bereaksi *setelah* krisis harga terjadi, bukan sebelumnya.

PanganWatch menjawab pertanyaan kunci:
- **Provinsi mana** yang paling rentan terhadap instabilitas harga pangan?
- **Bagaimana** pola supply produksi padi mempengaruhi harga beras di tingkat konsumen?
- **Kapan** harga beras berpotensi melonjak berdasarkan kondisi supply?

---

## 🚀 Demo

🔗 **Live App:** [panganwatch.streamlit.app](https://YOUR-APP-URL.streamlit.app)

![PanganWatch Dashboard](https://img.shields.io/badge/Status-Live-brightgreen)

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🗺️ **Clustering Provinsi** | Klasifikasi 38 provinsi ke 4 profil ketahanan pangan menggunakan K-Means |
| 🤖 **Prediksi Harga** | Model Random Forest untuk memprediksi harga beras berdasarkan supply nasional |
| 📈 **Tren Harga** | Visualisasi pergerakan harga 6 kualitas beras (Jan 2024 – Apr 2026) |
| 📊 **Supply Nasional** | Monitoring produksi, luas panen, dan yield bulanan 2025 |
| 🔮 **Simulasi Interaktif** | Input nilai supply → dapatkan estimasi harga beras |

---

## 🗃️ Dataset

| Dataset | Sumber | Periode |
|---------|--------|---------|
| Produksi Padi per Provinsi (Bulanan) | BPS | 2025 |
| Luas Panen Padi per Provinsi (Bulanan) | BPS | 2025 |
| Indeks Harga Konsumen Makanan | BPS | 2024–2026 |
| Harga Beras per Kualitas | Kemendag | 2024–2026 |

> Semua dataset bersumber dari data publik pemerintah Indonesia dan tidak dilindungi hak cipta.

---

## 🛠️ Tech Stack

```
Python 3.10+
├── pandas, numpy          → data processing
├── scikit-learn           → K-Means clustering, Random Forest
├── plotly                 → interactive visualization
├── streamlit              → web dashboard
└── Microsoft Azure        → Blob Storage (dataset & model artifact)
```

---

## 📁 Struktur Repo

```
panganwatch/
├── app.py                          # Streamlit dashboard utama
├── requirements.txt                # Python dependencies
├── output_harga_beras_clean.csv    # Data harga beras 2024-2026
├── output_clustering_provinsi.csv  # Hasil clustering 38 provinsi
├── output_supply_nasional.csv      # Agregasi supply nasional 2025
├── notebook/
│   └── datathon_ketahanan_pangan.ipynb  # EDA & model training
└── README.md
```

---

## ⚙️ Cara Menjalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/faeezzy/panganwatch.git
cd panganwatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan Streamlit
streamlit run app.py
```

---

## 📊 Hasil Analisis

### Clustering 38 Provinsi

| Cluster | Label | Provinsi | Karakteristik |
|---------|-------|----------|---------------|
| 1 | 🟢 Lumbung Padi | Jawa Barat, Jawa Tengah, Jawa Timur | Produksi > 9 juta ton, yield tinggi |
| 2 | 🔵 Produksi Tinggi | Aceh, Bali, Banten, NTB, dll | Produksi menengah-tinggi, yield stabil |
| 3 | 🟡 Produksi Menengah | Kalimantan, Sulawesi, Maluku, dll | Produksi menengah, perlu peningkatan |
| 4 | 🔴 Produksi Rendah | Kep. Riau, Papua, dll | Produksi rendah, volatilitas tinggi |

### Key Insights
- Harga beras naik **9.2%** dari Januari 2024 ke April 2026
- **3 provinsi** (Jawa Barat, Tengah, Timur) menyumbang mayoritas produksi padi nasional
- Provinsi kepulauan dan Papua menunjukkan **volatilitas yield tertinggi** — indikator kerentanan struktural
- Puncak produksi nasional terjadi pada **Maret** sesuai siklus musim tanam utama

---

## 🔮 Rencana Pengembangan

- [ ] Integrasi data cuaca BMKG sebagai fitur prediktif
- [ ] Perluasan ke komoditas lain (cabai, bawang, jagung)
- [ ] Early warning system otomatis berbasis anomali detection
- [ ] Prediksi rolling 3 bulan dengan confidence interval
- [ ] API endpoint untuk integrasi sistem pemerintah

---

## 👥 Tim

| Nama | Peran |
|------|-------|
| Faizah Zahra Aqilah | Data Scientist(aspiring) & Developer |

---

## 📄 Lisensi

Dataset bersumber dari data publik BPS dan Kemendag. Kode dalam repo ini menggunakan lisensi MIT.

---

<div align="center">
  <sub>Dibuat untuk Microsoft AI Impact Challenge — Tema Ketahanan Pangan & Agrikultur Modern</sub>
</div>
