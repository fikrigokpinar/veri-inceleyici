import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(page_title="Veri Kümesi İnceleyici", layout="wide")

st.title("📊 Veri Kümesi İnceleyici")

uploaded_file = st.file_uploader("Bir CSV veya Excel dosyası yükleyin", type=["csv", "xlsx"])

# Dosya varsa yükle, yoksa örnek veri ata
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Dosya başarıyla yüklendi.")
    except Exception as e:
        st.error(f"Dosya okunamadı: {e}")
        df = None
else:
    st.info("📂 Dosya yüklenmedi. Örnek veri kümesi yüklendi.")
    iris = load_iris(as_frame=True)
    df = iris.frame

if df is not None:
    st.subheader("📋 İlk 5 Gözlem")
    st.dataframe(df.head())

    st.subheader("📈 Özet İstatistikler")
    st.write(df.describe())

    st.subheader("🔢 Veri Kümesi Bilgisi")
    st.write(f"Gözlem sayısı: {df.shape[0]}")
    st.write(f"Değişken sayısı: {df.shape[1]}")
    st.write("Veri tipleri:")
    st.write(df.dtypes)

    # 🎨 Grafik bölümü
    st.subheader("📊 Grafiksel Görselleştirme")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 0:
        selected_col = st.selectbox("Bir sayısal değişken seçin:", num_cols)
        chart_type = st.radio("Grafik türü seçin:", ("Histogram", "Boxplot"))

        fig, ax = plt.subplots()
        if chart_type == "Histogram":
            ax.hist(df[selected_col].dropna(), bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f"{selected_col} - Histogram")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frekans")
        else:
            ax.boxplot(df[selected_col].dropna(), vert=False)
            ax.set_title(f"{selected_col} - Boxplot")
            ax.set_xlabel(selected_col)

        st.pyplot(fig)
    else:
        st.info("Grafik için uygun sayısal sütun bulunamadı.")
