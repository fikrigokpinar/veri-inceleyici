import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.datasets import (
    load_iris, load_breast_cancer, load_digits, fetch_california_housing,
    make_blobs, make_regression
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Fonksiyonlar
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

def compare_imputation_methods(df_full, df_with_nan, methods):
    mask = df_with_nan.isnull()
    numeric_cols = df_full.select_dtypes(include='number').columns
    results = []

    for name, imputer in methods.items():
        df_temp = df_with_nan.copy()
        df_temp[numeric_cols] = imputer.fit_transform(df_temp[numeric_cols])
        
        for col in numeric_cols:
            if mask[col].any():
                y_true = df_full.loc[mask[col], col]
                y_pred = df_temp.loc[mask[col], col]
                
                results.append({
                    "Yöntem": name,
                    "Değişken": col,
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "MSE": mean_squared_error(y_true, y_pred),
                    "MAPE (%)": mape(y_true, y_pred)
                })

    return pd.DataFrame(results)

st.set_page_config(page_title="Veri Kümesi İnceleyici", layout="wide")
st.title("📊 Veri Kümesi İnceleyici")

uploaded_file = st.file_uploader("Bir CSV veya Excel dosyası yükleyin", type=["csv", "xlsx"])

# -------------------- VERİ SEÇİMİ --------------------
import numpy as np  # En başta olmalı
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



# -------------------- VERİ YÜKLEME --------------------# Eğer kullanıcı veri yüklemediyse, örnek veri kullanalım
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
    

add_missing = st.checkbox("🔥 Yapay eksik veri ekle", value=True)
missing_ratio = st.selectbox("Eksik veri oranı (p)", [0.05, 0.10, 0.15, 0.20], index=0, format_func=lambda x: f"%{int(x*100)}")

df = get_builtin_dataset(dataset_name)
df_clean = df.copy()  # Tam hali

if add_missing and not df.empty:
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    n_missing = int(total_cells * missing_ratio)

    st.write(f"📉 Toplam {total_cells} hücrede, yaklaşık **{n_missing}** adet eksik değer eklenecek.")

    # Eksik değerleri yerleştirmek için rastgele hücre seç
    for _ in range(n_missing):
        i = np.random.randint(0, n_rows)
        j = np.random.randint(0, n_cols)
        df.iat[i, j] = np.nan
# Eksik veri eklendikten hemen sonra:

st.session_state["df_complete"] = df_clean
st.session_state["df_with_nan"] = df.copy()




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
    # -------------------- 🎯 Hedef Seçimi + Grafik --------------------

st.subheader("🎯 Hedef ve Bağımsız Değişken Analizi")

# Otomatik öneri (son sütun)
guessed_target = df.columns[-1]
st.info(f"⚠️ Önerilen hedef değişken: `{guessed_target}` (değiştirebilirsiniz)")

# Hedef değişken seçimi
target = st.selectbox("Hedef değişkeni (H) seçin:", df.columns, index=len(df.columns) - 1)

# Hedef dışındaki sütunları seçtir
remaining_cols = [col for col in df.columns if col != target]
selected_b = st.selectbox("Bağımsız değişken (B) seçin:", remaining_cols)

# Tür kontrolü
target_type = 'S' if pd.api.types.is_numeric_dtype(df[target]) else 'K'
b_type = 'S' if pd.api.types.is_numeric_dtype(df[selected_b]) else 'K'

fig, ax = plt.subplots()

if target_type == 'S' and b_type == 'S':
    st.markdown("📌 H ve B ikisi de **sürekli** → Scatterplot")
    ax.scatter(df[selected_b], df[target], alpha=0.6)
    ax.set_xlabel(selected_b)
    ax.set_ylabel(target)
    ax.set_title(f"{selected_b} vs {target}")

elif target_type == 'S' and b_type == 'K':
    st.markdown("📌 H sürekli, B kategorik → Kategorilere göre Boxplot")
    df.boxplot(column=target, by=selected_b, ax=ax)
    ax.set_title(f"{target} by {selected_b}")
    plt.suptitle('')

elif target_type == 'K' and b_type == 'S':
    st.markdown("📌 H kategorik, B sürekli → Kategorilere göre Boxplot")
    df.boxplot(column=selected_b, by=target, ax=ax)
    ax.set_title(f"{selected_b} by {target}")
    plt.suptitle('')

elif target_type == 'K' and b_type == 'K':
    st.markdown("📌 H ve B ikisi de **kategorik** → Gruplu çubuk grafik")
    counts = df.groupby([selected_b, target]).size().unstack().fillna(0)
    counts.plot(kind='bar', ax=ax)
    ax.set_ylabel("Frekans")
    ax.set_title(f"{selected_b} ve {target} dağılımı")
    st.pyplot(fig)
    st.stop()
else:
    st.warning("Bu kombinasyon için uygun grafik belirlenemedi.")

st.pyplot(fig)

# -------------------- 📊 Bağımsız Değişkenler Arası Grafik --------------------

st.subheader("📊 Bağımsız Değişkenler Arası Görselleştirme")

# Hedef değişkeni hariç tüm sütunlar
feature_cols = [col for col in df.columns if col != target]

if len(feature_cols) < 2:
    st.warning("En az iki bağımsız değişken gerekli.")
else:
    b1 = st.selectbox("Bağımsız değişken 1 (B₁):", feature_cols, key="b1_select")
    remaining = [c for c in feature_cols if c != b1]
    b2 = st.selectbox("Bağımsız değişken 2 (B₂):", remaining, key="b2_select")

    b1_type = 'S' if pd.api.types.is_numeric_dtype(df[b1]) else 'K'
    b2_type = 'S' if pd.api.types.is_numeric_dtype(df[b2]) else 'K'

    fig, ax = plt.subplots()

    if b1_type == 'S' and b2_type == 'S':
        st.markdown("📌 B₁ ve B₂ ikisi de **sürekli** → Scatterplot")
        ax.scatter(df[b1], df[b2], alpha=0.6)
        ax.set_xlabel(b1)
        ax.set_ylabel(b2)
        ax.set_title(f"{b1} vs {b2}")

    elif b1_type == 'K' and b2_type == 'S':
        st.markdown("📌 B₁ kategorik, B₂ sürekli → Kategorilere göre Boxplot")
        df.boxplot(column=b2, by=b1, ax=ax)
        ax.set_title(f"{b2} by {b1}")
        plt.suptitle('')

    elif b1_type == 'S' and b2_type == 'K':
        st.markdown("📌 B₁ sürekli, B₂ kategorik → Kategorilere göre Boxplot")
        df.boxplot(column=b1, by=b2, ax=ax)
        ax.set_title(f"{b1} by {b2}")
        plt.suptitle('')

    elif b1_type == 'K' and b2_type == 'K':
        st.markdown("📌 B₁ ve B₂ ikisi de **kategorik** → Gruplu çubuk grafik")
        counts = df.groupby([b1, b2]).size().unstack().fillna(0)
        counts.plot(kind='bar', ax=ax)
        ax.set_ylabel("Frekans")
        ax.set_title(f"{b1} ve {b2} dağılımı")
        st.pyplot(fig)
        st.stop()
    else:
        st.warning("Bu kombinasyon için uygun grafik belirlenemedi.")

    st.pyplot(fig)
# -------------------- 📋 Kategorik Değişken Frekans Tabloları (Manuel Seçim Dahil) --------------------

st.subheader("📋 Kategorik Değişken Frekans Tabloları")

# Otomatik öneri (nunique ≤ 20, float olmayanlar)
auto_cat_candidates = [
    col for col in df.columns
    if df[col].nunique() <= 20 and not pd.api.types.is_float_dtype(df[col])
]

st.info(f"⚙️ Otomatik önerilen kategorik değişkenler: {', '.join(auto_cat_candidates) if auto_cat_candidates else 'Yok'}")

# Tüm değişkenleri seçilebilir hale getir (kullanıcı isterse override etsin)
selected_cat = st.selectbox("Frekans analizi için bir değişken seçin:", df.columns, key="cat_freq_any")

# Frekans tablosunu oluştur ve göster
freq_table = df[selected_cat].value_counts().reset_index()
freq_table.columns = [selected_cat, "Frekans"]

st.write("📊 Frekans Tablosu")
st.dataframe(freq_table)

# Grafikle göster
fig, ax = plt.subplots()
freq_table.plot(kind='bar', x=selected_cat, y="Frekans", ax=ax, legend=False, color='orange', edgecolor='black')
ax.set_title(f"{selected_cat} - Frekans Dağılımı")
ax.set_ylabel("Frekans")
ax.set_xlabel(selected_cat)
st.pyplot(fig)
# -------------------- 🔥 Eksik Veri Analizi --------------------

import seaborn as sns

st.subheader("🔥 Eksik Veri Analizi (Isı Haritası ve Tablo)")

missing_counts = df.isnull().sum()
total_missing = missing_counts.sum()

if total_missing == 0:
    st.success("Veri kümesinde eksik gözlem yok ✅")
else:
    # Eksik değer özeti
    st.write("📋 Eksik Değer Tablosu")
    missing_df = pd.DataFrame({
        "Değişken": missing_counts.index,
        "Eksik Sayısı": missing_counts.values,
        "Eksik Oranı (%)": (missing_counts.values / len(df)) * 100
    })
    missing_df = missing_df[missing_df["Eksik Sayısı"] > 0]
    st.dataframe(missing_df)

    # Eksik veri ısı haritası
    st.write("🗺️ Eksik Değer Isı Haritası")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Veri Kümesinde Eksik Gözlem Haritası")
    st.pyplot(fig)

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# -----------------------------
# Streamlit tarafında seçim menüsü
st.subheader("📌 Eksik Veri Doldurma")

impute_method = st.selectbox("Bir doldurma yöntemi seçin:", [
    "🔹 Basit: Ortalama (Mean)",
    "🔹 Basit: Medyan (Median)",
    "🔹 Basit: Mod (Mode)",
    "🔹 Basit: Forward Fill (ffill)",
    "🔹 Basit: Backward Fill (bfill)",
    "🔹 Basit: Doğrusal Enterpolasyon",
    "🔸 Kural: kNN Imputation",
    "🔸 Kural: Iterative Imputer",
    "🔸 Kural: Regresyon Temelli",
    "🟢 ML: Random Forest",
    "🟢 ML: XGBoost",
    "🟢 ML: LightGBM"
])

# -----------------------------
# Eksik veri doldurma fonksiyonu

def impute_data(df, method, target_col=None):
    df_copy = df.copy()

    if target_col:
        target_series = df_copy[target_col]
        df_copy = df_copy.drop(columns=[target_col])
    else:
        target_series = None

    if method == "🔹 Basit: Ortalama (Mean)":
        df_copy = df_copy.fillna(df_copy.mean(numeric_only=True))

    elif method == "🔹 Basit: Medyan (Median)":
        df_copy = df_copy.fillna(df_copy.median(numeric_only=True))

    elif method == "🔹 Basit: Mod (Mode)":
        df_copy = df_copy.fillna(df_copy.mode().iloc[0])

    elif method == "🔹 Basit: Forward Fill (ffill)":
        df_copy = df_copy.fillna(method='ffill')

    elif method == "🔹 Basit: Backward Fill (bfill)":
        df_copy = df_copy.fillna(method='bfill')

    elif method == "🔹 Basit: Doğrusal Enterpolasyon":
        df_copy = df_copy.interpolate()

    elif method == "🔸 Kural: kNN Imputation":
        imputer = KNNImputer(n_neighbors=5)
        df_copy[df_copy.select_dtypes(include=['number']).columns] = imputer.fit_transform(
            df_copy.select_dtypes(include=['number']))

    elif method == "🔸 Kural: Iterative Imputer":
        imputer = IterativeImputer()
        df_copy[df_copy.select_dtypes(include=['number']).columns] = imputer.fit_transform(
            df_copy.select_dtypes(include=['number']))

    elif method in ["🔸 Kural: Regresyon Temelli", "🟢 ML: Random Forest", "🟢 ML: XGBoost", "🟢 ML: LightGBM"]:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor

        model_map = {
            "🔸 Kural: Regresyon Temelli": LinearRegression(),
            "🟢 ML: Random Forest": RandomForestRegressor(),
            "🟢 ML: XGBoost": XGBRegressor(),
            "🟢 ML: LightGBM": LGBMRegressor()
        }

        for col in df_copy.columns:
            if df_copy[col].isnull().sum() > 0:
                not_null = df_copy[df_copy[col].notnull()]
                is_null = df_copy[df_copy[col].isnull()]
                X_train = not_null.drop(columns=[col]).select_dtypes(include=['number'])
                y_train = not_null[col]
                X_pred = is_null.drop(columns=[col]).select_dtypes(include=['number'])

                if not X_pred.empty and not X_train.empty:
                    model = model_map[method]
                    model.fit(X_train, y_train)
                    df_copy.loc[df_copy[col].isnull(), col] = model.predict(X_pred)

    # Hedefi geri ekle
    if target_series is not None:
        df_copy[target_col] = target_series

    return df_copy


# -----------------------------
# Uygulama: Eğer eksik veri varsa kullanıcıdan seçim al, uygula

if df.isnull().sum().sum() > 0:
    if st.button("Eksik Verileri Doldur"):
        df_before = df.copy()
        df = impute_data(df, impute_method, target_col=target)
        df_after = df.copy()
        st.session_state["df_after"] = df_after
        st.session_state["df_before"] = df_before
        st.success("✔ Eksik veriler dolduruldu.")

import matplotlib.pyplot as plt
import seaborn as sns

# Karşılaştırmalı grafikler için önce veri kontrolü
# Karşılaştırmalı grafikler için önce veri kontrolü
if "df_after" in st.session_state and "df_before" in st.session_state:
    df_after = st.session_state["df_after"]
    df_before = st.session_state["df_before"]

    st.subheader("📊 Eksik Veri Doldurma Öncesi ve Sonrası Karşılaştırma")

    # Ortak sayısal sütunları al
    numeric_cols = df_after.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col in df_before.columns]

    if numeric_cols:
        selected_col = st.selectbox("Karşılaştırmak istediğiniz sayısal sütunu seçin:", numeric_cols)
        if selected_col in df_before.columns:

            mask_missing = df_before[selected_col].isnull()
            filled_values = df_after.loc[mask_missing, selected_col]

            if not filled_values.empty:
                # Boxplot
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=pd.DataFrame({
                    "Önce": df_before[selected_col].dropna(),
                    "Sonra": df_after[selected_col]
                }), orient="v", ax=ax)
                ax.set_title(f"{selected_col} - Eksik Veri Doldurma Öncesi ve Sonrası Boxplot")
                st.pyplot(fig)

            # Histogram
                fig2, ax2 = plt.subplots()
                sns.histplot(df_before[selected_col].dropna(), color='blue', label='Önce', kde=True, stat='density')
                sns.histplot(df_after[selected_col], color='orange', label='Sonra', kde=True, stat='density')
                ax2.legend()
                ax2.set_title("Histogram: Eksik Veri Doldurma Öncesi ve Sonrası")
                st.pyplot(fig2)

                # Scatter (diğer bir değişkene göre)
                other_cols = [col for col in numeric_cols if col != selected_col]
            if other_cols:
                other_col = other_cols[0]
                fig3, ax3 = plt.subplots()
                ax3.scatter(df_before[other_col], df_before[selected_col], alpha=0.5, label='Önce')
                ax3.scatter(df_after[other_col], df_after[selected_col], alpha=0.5, label='Sonra')
                ax3.set_xlabel(other_col)
                ax3.set_ylabel(selected_col)
                ax3.set_title("Dağılım Grafiği (Scatter): Önce vs Sonra")
                ax3.legend()
                st.pyplot(fig3)
        else:
            st.info("Bu değişkende eksik veri yoktu, karşılaştırma yapılacak veri bulunamadı.")
    else:
        st.warning("Karşılaştırma için uygun sayısal değişken bulunamadı.")
else:
    st.info("Eksik veri doldurulmadan önce-sonra karşılaştırması yapılamaz.")

# -------------------- 📈 Eksik Veri Doldurma Yöntemi Karşılaştırması --------------------
if "df_complete" in st.session_state and "df_with_nan" in st.session_state:
    st.subheader("📈 Eksik Veri Doldurma Yöntemlerinin Karşılaştırılması")

    from sklearn.impute import SimpleImputer

    methods = {
        "Ortalama": SimpleImputer(strategy="mean"),
        "Medyan": SimpleImputer(strategy="median"),
        "KNN": KNNImputer(n_neighbors=3),
        "Iterative (BayesianRidge)": IterativeImputer(random_state=0),
        "Random Forest": IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=0), random_state=0),
        "XGBoost": IterativeImputer(estimator=XGBRegressor(n_estimators=10, verbosity=0, random_state=0), random_state=0)
    }

    df_full = st.session_state["df_complete"]
    df_with_nan = st.session_state["df_with_nan"]

    with st.spinner("Yöntemler test ediliyor..."):
        comparison_df = compare_imputation_methods(df_full, df_with_nan, methods)
        summary = comparison_df.groupby("Yöntem")[["MAE", "MSE", "MAPE (%)"]].mean().sort_values("MAE")

    st.markdown("### 📋 Detaylı Karşılaştırma")
    st.dataframe(comparison_df)

    st.markdown("### 🏆 Ortalama Performans Karşılaştırması")
    st.dataframe(summary)
else:
    st.info("Bu karşılaştırma yalnızca örnek veri kümesinde eksik değerler yapay olarak eklendiğinde mümkündür.")

