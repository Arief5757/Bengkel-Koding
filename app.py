import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    page_title="Prediksi Kategori Obesitas",
    page_icon="🏋️",
    layout="centered"
)

# Label mapping
label_mapping = {
    0: "Berat Badan Kurang",
    1: "Berat Badan Normal",
    2: "Kelebihan Berat Badan Tingkat I",
    3: "Kelebihan Berat Badan Tingkat II",
    4: "Obesitas Tipe I",
    5: "Obesitas Tipe II",
    6: "Obesitas Tipe III"
}

# Load models with caching
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()

# App header
st.title("🏋️ Prediksi Kategori Berat Badan")
st.markdown("Aplikasi ini memprediksi kategori berat badan berdasarkan parameter kesehatan Anda.")

# Input form
with st.form("prediction_form"):
    st.subheader("Data Diri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox(
            "Jenis Kelamin",
            ["Laki-laki", "Perempuan"],
            index=None,
            placeholder="Pilih jenis kelamin"
        )
        weight = st.number_input(
            "Berat Badan (kg)",
            min_value=0.0,
            step=0.1,
            help="Masukkan berat badan dalam kilogram"
        )
    
    with col2:
        age = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=120,
            step=1
        )
        height_cm = st.number_input(
            "Tinggi Badan (cm)",
            min_value=0.0,
            step=0.1,
            help="Masukkan tinggi badan dalam centimeter"
        )
    
    family_history = st.selectbox(
        "Riwayat Keluarga Obesitas",
        ["Ya", "Tidak"],
        index=None,
        placeholder="Apakah ada riwayat obesitas dalam keluarga?"
    )
    
    submitted = st.form_submit_button(
        "Prediksi Kategori",
        type="primary",
        use_container_width=True
    )

# Prediction logic
if submitted:
    if None in [gender, family_history] or weight <= 0 or height_cm <= 0:
        st.warning("Harap lengkapi semua data dengan benar!")
    else:
        with st.spinner('Sedang menganalisis...'):
            # Process inputs
            height = height_cm / 100
            gender_val = 1 if gender == "Laki-laki" else 0
            family_val = 1 if family_history == "Ya" else 0

            # Create input DataFrame
            input_df = pd.DataFrame([[gender_val, weight, height, age, family_val]], 
                                  columns=['Gender', 'Weight', 'Height', 'Age', 'family_history_with_overweight'])

            # Scale and predict
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)
            
            # Get result
            result_label = label_mapping.get(int(pred[0]), "Tidak diketahui")
            
            # Calculate BMI
            bmi = weight / (height ** 2)
            
            # Display results
            st.subheader("Hasil Prediksi")
            
            # Create metric cards
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Indeks Massa Tubuh (BMI)", f"{bmi:.1f}")
            with col2:
                st.metric("Kategori", result_label)
            
            # Add color-coded interpretation
            st.subheader("Interpretasi")
            if "Kurang" in result_label:
                st.info("""
                🔹 **Rekomendasi**:
                - Pertimbangkan untuk menambah asupan kalori sehat
                - Konsultasikan dengan ahli gizi untuk rencana makan
                - Lakukan latihan kekuatan untuk membangun massa otot
                """)
            elif "Normal" in result_label:
                st.success("""
                ✅ **Rekomendasi**:
                - Pertahankan pola makan sehat dan seimbang
                - Lakukan aktivitas fisik secara teratur
                - Pantau berat badan secara berkala
                """)
            elif "Kelebihan" in result_label:
                st.warning("""
                ⚠️ **Rekomendasi**:
                - Pertimbangkan untuk mengurangi asupan kalori
                - Tingkatkan aktivitas fisik harian
                - Konsultasikan dengan dokter untuk pemeriksaan
                """)
            elif "Obesitas" in result_label:
                st.error("""
                ❗ **Rekomendasi**:
                - Sangat disarankan berkonsultasi dengan dokter
                - Pertimbangkan program penurunan berat badan profesional
                - Fokus pada perubahan gaya hidup jangka panjang
                """)

# Footer
st.markdown("---")
st.caption("""
    Aplikasi prediksi ini bersifat informatif dan tidak menggantikan konsultasi medis profesional. 
    Hasil mungkin bervariasi tergantung pada faktor individu lainnya.
""")