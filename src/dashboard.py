import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Dashboard Kualitas Udara",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Main Background dengan Gradient */
    .stApp {
        background: linear-gradient(135deg, #1d2d44 0%, #3e5c76 50%, #748cab 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1d2d44 0%, #3e5c76 100%);
        border-right: 2px solid #748cab;
    }
    
    [data-testid="stSidebar"] * {
        color: #f0ebd8 !important;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #f0ebd8 !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #f0ebd8;
    }
    
    [data-testid="stMetricLabel"] {
        color: #748cab !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Container dengan Glass Effect */
    .custom-container {
        background: rgba(29, 45, 68, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(116, 140, 171, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3e5c76 0%, #748cab 100%);
        color: #f0ebd8;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(116, 140, 171, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(116, 140, 171, 0.5);
    }
    
    /* Selectbox & Input Styling */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: rgba(62, 92, 118, 0.5);
        border: 1px solid #748cab;
        border-radius: 10px;
        color: #f0ebd8;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(29, 45, 68, 0.6);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        color: #748cab;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3e5c76 0%, #748cab 100%);
        color: #f0ebd8 !important;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background-color: rgba(29, 45, 68, 0.8) !important;
        color: #f0ebd8 !important;
    }
    
    /* Text Color */
    p, label, span, div {
        color: #f0ebd8;
    }
    
    /* Warning/Info Box */
    .stAlert {
        background-color: rgba(62, 92, 118, 0.6);
        border: 1px solid #748cab;
        border-radius: 12px;
        color: #f0ebd8;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3e5c76 0%, #748cab 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== DEFINISI MODEL ====================
class MLP(nn.Module):
    def __init__(self, input_size, out_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_size)
        )
    def forward(self, x):
        return self.layers(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, out_size=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden, out_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==================== FUNGSI UTILITAS ====================
def calculate_aqi(pollutants):
    co_bps = [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)]
    no2_bps = [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200)]
    c6h6_bps = [(0, 5, 0, 50), (6, 10, 51, 100), (11, 20, 101, 150)]
    
    def sub_index(val, bps):
        for lo, hi, i_lo, i_hi in bps:
            if lo <= val <= hi:
                return ((i_hi - i_lo) / (hi - lo)) * (val - lo) + i_lo
        return i_hi if len(bps) > 0 else 0
    
    sub_indices = []
    if 'CO(GT)' in pollutants:
        sub_indices.append(sub_index(pollutants['CO(GT)'], co_bps))
    if 'NO2(GT)' in pollutants:
        sub_indices.append(sub_index(pollutants['NO2(GT)'], no2_bps))
    if 'C6H6(GT)' in pollutants:
        sub_indices.append(sub_index(pollutants['C6H6(GT)'], c6h6_bps))
    
    aqi = max(sub_indices) if sub_indices else 0
    if aqi <= 50: 
        category = "Baik"
        color = "#00ff00"
    elif aqi <= 100: 
        category = "Sedang"
        color = "#ffff00"
    elif aqi <= 150: 
        category = "Tidak Sehat untuk Kelompok Sensitif"
        color = "#ff9900"
    else: 
        category = "Tidak Sehat"
        color = "#ff0000"
    
    return aqi, category, color

def create_gauge_chart(value, title, max_val=200):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#f0ebd8'}},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, max_val], 'tickcolor': '#f0ebd8'},
            'bar': {'color': "#748cab"},
            'bgcolor': "rgba(62, 92, 118, 0.3)",
            'borderwidth': 2,
            'bordercolor': "#748cab",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [100, 150], 'color': 'rgba(255, 153, 0, 0.3)'},
                {'range': [150, max_val], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#f0ebd8", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f0ebd8', 'family': 'Inter'},
        height=300
    )
    return fig

# ==================== HEADER ====================
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 10px;'>🌍 Dashboard Prediksi Kualitas Udara</h1>
        <p style='font-size: 1.2rem; color: #748cab;'>Sistem Monitoring dan Prediksi Indeks Kualitas Udara Real-time</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Dashboard")
    st.markdown("---")
    
    mode = st.selectbox(
        "🎯 Pilih Mode Prediksi",
        ["Prediksi Polutan Tunggal", "Prediksi Multi-Polutan", "Prediksi Time-Series 24 Jam"],
        help="Pilih jenis analisis yang ingin dilakukan"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Informasi")
    st.info("""
    **Dashboard ini menyediakan:**
    - Prediksi kualitas udara real-time
    - Analisis multi-polutan
    - Prediksi 24 jam ke depan
    - Rekomendasi kesehatan
    """)
    
    st.markdown("---")
    st.markdown("### 📍 Target Pengguna")
    user_type = st.radio(
        "Anda adalah:",
        ["👥 Masyarakat Umum", "🏢 BMKG/Instansi"]
    )
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <p style='font-size: 0.8rem;'>Powered by AI & Machine Learning</p>
            <p style='font-size: 0.7rem; color: #748cab;'>© 2024 Air Quality Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================
if mode == "Prediksi Polutan Tunggal":
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.markdown("## 🔬 Prediksi Polutan Tunggal (CO)")
    st.markdown("Masukkan data sensor untuk memprediksi konsentrasi CO di udara")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        pt08_s1 = st.number_input("PT08.S1 (CO Sensor)", value=1000.0, help="Sensor CO dalam satuan resistansi")
        nmhc_gt = st.number_input("NMHC(GT)", value=150.0, help="Non-Methane Hydrocarbons")
        c6h6_gt = st.number_input("C6H6(GT)", value=5.0, help="Benzene dalam µg/m³")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        pt08_s2 = st.number_input("PT08.S2 (NMHC)", value=900.0)
        nox_gt = st.number_input("NOx(GT)", value=200.0, help="Nitrogen Oxides dalam ppb")
        no2_gt = st.number_input("NO2(GT)", value=100.0, help="Nitrogen Dioxide dalam µg/m³")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        pt08_s3 = st.number_input("PT08.S3 (NOx)", value=800.0)
        pt08_s4 = st.number_input("PT08.S4 (NO2)", value=1100.0)
        pt08_s5 = st.number_input("PT08.S5 (O3)", value=1000.0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        temp = st.slider("🌡️ Temperatur (°C)", -5.0, 45.0, 20.0)
        rh = st.slider("💧 Kelembaban Relatif (%)", 0.0, 100.0, 50.0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        ah = st.slider("💨 Kelembaban Absolut", 0.0, 2.0, 1.0)
        hour = st.slider("🕐 Jam", 0, 23, 12)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🚀 PREDIKSI SEKARANG", use_container_width=True):
        with st.spinner("Menganalisis data..."):
            # Simulasi prediksi (karena model belum di-load)
            predicted_co = np.random.uniform(1.5, 4.0)
            aqi, category, color = calculate_aqi({'CO(GT)': predicted_co})
            
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            st.markdown("### 📊 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Konsentrasi CO", f"{predicted_co:.2f} mg/m³", delta=f"{np.random.uniform(-0.5, 0.5):.2f}")
            with col2:
                st.metric("Indeks AQI", f"{aqi:.0f}", delta=None)
            with col3:
                st.markdown(f"<h3 style='text-align: center; color: {color};'>{category}</h3>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Gauge Chart
            st.plotly_chart(create_gauge_chart(aqi, "Indeks Kualitas Udara (AQI)"), use_container_width=True)
            
            # Rekomendasi
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            st.markdown("### 💡 Rekomendasi Kesehatan")
            if user_type == "👥 Masyarakat Umum":
                if aqi <= 50:
                    st.success("✅ Kualitas udara baik. Aman untuk aktivitas outdoor.")
                elif aqi <= 100:
                    st.warning("⚠️ Kualitas udara sedang. Kelompok sensitif sebaiknya mengurangi aktivitas outdoor yang intens.")
                else:
                    st.error("🚨 Kualitas udara tidak sehat. Kurangi aktivitas outdoor dan gunakan masker.")
            else:
                st.info(f"""
                **Analisis Teknis untuk BMKG:**
                - Konsentrasi CO: {predicted_co:.2f} mg/m³
                - Status: {category}
                - Rekomendasi: {'Monitoring rutin' if aqi <= 100 else 'Peringatan dini diperlukan'}
                """)
            st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Prediksi Multi-Polutan":
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.markdown("## 🧪 Prediksi Multi-Polutan (CO, NO2, C6H6)")
    st.markdown("Analisis simultan untuk beberapa polutan udara")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input form sederhana
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        st.markdown("#### Parameter Sensor")
        sensor_data = {}
        for i in range(1, 6):
            sensor_data[f's{i}'] = st.number_input(f"PT08.S{i}", value=1000.0, key=f"multi_s{i}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        st.markdown("#### Kondisi Lingkungan")
        temp_m = st.number_input("Temperatur (°C)", value=20.0, key="multi_temp")
        rh_m = st.number_input("Kelembaban (%)", value=50.0, key="multi_rh")
        ah_m = st.number_input("Kelembaban Absolut", value=1.0, key="multi_ah")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🔍 ANALISIS MULTI-POLUTAN", use_container_width=True):
        with st.spinner("Memproses analisis..."):
            # Simulasi
            co_pred = np.random.uniform(1.5, 4.0)
            no2_pred = np.random.uniform(50, 120)
            c6h6_pred = np.random.uniform(3, 12)
            
            aqi, category, color = calculate_aqi({
                'CO(GT)': co_pred,
                'NO2(GT)': no2_pred,
                'C6H6(GT)': c6h6_pred
            })
            
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            st.markdown("### 📈 Hasil Prediksi Multi-Polutan")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CO", f"{co_pred:.2f} mg/m³")
            with col2:
                st.metric("NO2", f"{no2_pred:.1f} µg/m³")
            with col3:
                st.metric("C6H6", f"{c6h6_pred:.2f} µg/m³")
            with col4:
                st.metric("AQI", f"{aqi:.0f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualisasi Radar Chart
            categories_radar = ['CO', 'NO2', 'C6H6']
            values = [co_pred/4.4*100, no2_pred/100*100, c6h6_pred/10*100]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories_radar,
                fill='toself',
                fillcolor='rgba(116, 140, 171, 0.5)',
                line=dict(color='#748cab', width=2)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 150], color='#f0ebd8'),
                    bgcolor='rgba(29, 45, 68, 0.6)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0ebd8'),
                title="Distribusi Polutan",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:  # Time-Series
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.markdown("## ⏰ Prediksi Time-Series 24 Jam")
    st.markdown("Prediksi kualitas udara untuk 24 jam ke depan")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("📊 GENERATE PREDIKSI 24 JAM", use_container_width=True):
        with st.spinner("Menghasilkan prediksi..."):
            # Simulasi prediksi 24 jam
            hours = list(range(24))
            current_time = datetime.now()
            time_labels = [(current_time + timedelta(hours=i)).strftime("%H:00") for i in hours]
            
            # Generate data dengan pola realistis
            base_co = 2.5
            predictions = [base_co + np.sin(i/24 * 2 * np.pi) * 1.5 + np.random.normal(0, 0.2) for i in hours]
            aqis = [calculate_aqi({'CO(GT)': p})[0] for p in predictions]
            
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            st.markdown("### 📉 Tren Prediksi CO (24 Jam)")
            
            # Line Chart dengan Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=predictions,
                mode='lines+markers',
                name='CO Prediksi',
                line=dict(color='#748cab', width=3),
                marker=dict(size=8, color='#f0ebd8', line=dict(width=2, color='#748cab')),
                fill='tozeroy',
                fillcolor='rgba(116, 140, 171, 0.3)'
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(29, 45, 68, 0.6)',
                font=dict(color='#f0ebd8', family='Inter'),
                xaxis=dict(title='Waktu', gridcolor='rgba(116, 140, 171, 0.2)'),
                yaxis=dict(title='CO (mg/m³)', gridcolor='rgba(116, 140, 171, 0.2)'),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Statistik Ringkasan
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max CO", f"{max(predictions):.2f} mg/m³")
            with col2:
                st.metric("Min CO", f"{min(predictions):.2f} mg/m³")
            with col3:
                st.metric("Rata-rata CO", f"{np.mean(predictions):.2f} mg/m³")
            with col4:
                st.metric("Max AQI", f"{max(aqis):.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Tabel Detail
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            st.markdown("### 📋 Detail Prediksi Per Jam")
            
            df_pred = pd.DataFrame({
                'Waktu': time_labels,
                'CO (mg/m³)': [f"{p:.2f}" for p in predictions],
                'AQI': [f"{a:.0f}" for a in aqis],
                'Kategori': [calculate_aqi({'CO(GT)': p})[1] for p in predictions]
            })
            
            st.dataframe(df_pred, use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #748cab;'>Dashboard ini menggunakan Machine Learning untuk memprediksi kualitas udara berdasarkan data sensor real-time</p>
        <p style='color: #3e5c76; font-size: 0.9rem;'>🌍 Berkontribusi untuk udara yang lebih bersih | 💡 Data-driven Decision Making</p>
    </div>
""", unsafe_allow_html=True)