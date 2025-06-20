import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_digits, fetch_california_housing,
    make_blobs, make_regression
)

st.set_page_config(page_title="Veri Kümesi İnceleyici", layout="wide")
st.title("📊 Veri Kümesi İnceleyici")

uploaded_file = st.file_uploader("Bir CSV veya Excel dosyası yükleyin", type=["csv", "xlsx"])

# -------------------- VERİ SEÇİMİ --------------------
def get_builtin_dataset(name):
    if name == "Iris (çoklu sınıflandırma)":
        return load_iris(as_frame=True).frame
    elif name == "Breast Cancer (ikili sınıflandırma)":
        return load_breast_cancer(as_frame=True).frame
    elif name == "Digits (0-9 sınıflandırma)":
        return load_digits(as_frame=True).frame
    elif name == "California Housing (regresyon)":
        return fetch_california_housing(as_frame=True).frame
    elif name == "Blobs (kümeleme)":
        X, y = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
        return pd.DataFrame(X, columns=["X1", "X2"]).assign(cluster=y)
    elif name == "Make Regression (yapay veri)":
        X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
        return pd.DataFrame({"X": X.flatten(), "Y": y})
    else:
        return pd.DataFrame()

# -------------------- VERİ YÜKLEME --------------------
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
    st.info("📂 Dosya yüklenmedi. Aşağıdan örnek veri seçebilirsiniz.")
    dataset_name = st.selectbox("📌 Örnek bir veri kümesi seçin:", [
        "Iris (çoklu sınıflandırma)",
        "Breast Cancer (ikili sınıflandırma)",
        "Digits (0-9 sınıflandırma)",
        "California Housing (regresyon)",
        "Blobs (kümeleme)",
        "Make Regression (yapay veri)"
    ])
    df = get_builtin_dataset(dataset_name)

# -------------------- ANALİZ --------------------
if df is not None and not df.empty:
    st.subheader("📋 İlk 5 Gözlem")
    st.dataframe(df.head())

    st.subheader("📈 Özet İstatistikler")
    st.write(df.describe())

    st.subheader("🔢 Veri Kümesi Bilgisi")
    st.write(f"Gözlem sayısı: {df.shape[0]}")
    st.write(f"Değişken sayısı: {df.shape[1]}")
    st.write("Veri tipleri:")
    st.write(df.dtypes)

    # -------------------- GRAFİK --------------------
    st.subheader("📊 Grafiksel Görselleştirme")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 0:
        selected_col = st.selectbox("Bir sayısal değişken seçin:", num_cols)
        chart_type = st.radio("Grafik türü seçin:", ("Histogram", "Boxplot"))

        fig, ax = plt.subplots()
        if chart_type == "Histogram":
            ax.hist(df[selected_col].dropna(), bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f"{selected_col} - Histogram")
        else:
            ax.boxplot(df[selected_col].dropna(), vert=False)
            ax.set_title(f"{selected_col} - Boxplot")
        st.pyplot(fig)
    else:
        st.info("Grafik için uygun sayısal sütun bulunamadı.")
