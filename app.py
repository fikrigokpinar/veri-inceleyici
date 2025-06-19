import streamlit as st
import pandas as pd

st.set_page_config(page_title="Veri Kümesi İnceleyici", layout="wide")

st.title("📊 Veri Kümesi İnceleyici")

# Kullanıcıdan dosya yüklemesini iste
uploaded_file = st.file_uploader("Bir CSV veya Excel dosyası yükleyin", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 İlk 5 Gözlem")
        st.dataframe(df.head())

        st.subheader("📈 Özet İstatistikler")
        st.write(df.describe())

        st.subheader("🔢 Veri Kümesi Bilgisi")
        st.write(f"Gözlem sayısı: {df.shape[0]}")
        st.write(f"Değişken sayısı: {df.shape[1]}")
        st.write("Veri tipleri:")
        st.write(df.dtypes)

    except Exception as e:
        st.error(f"Veri okunurken bir hata oluştu: {e}")
else:
    st.info("Lütfen bir veri dosyası yükleyin.")
