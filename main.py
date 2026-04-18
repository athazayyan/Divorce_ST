import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Divorce Prediction - DPS Pipeline",
    page_icon=None,
    layout="wide"
)

# ─────────────────────────────────────────────
# ATTRIBUTE DEFINITIONS
# ─────────────────────────────────────────────
ATTRIBUTES = {
    "Atr1":  "When I need it, I can take my discussions with my husband/wife from the beginning and correct it",
    "Atr2":  "When I argue with my husband/wife, it will eventually work for me to contact him/her",
    "Atr3":  "The time I spent with my husband/wife is special for me",
    "Atr4":  "Rather than being family, we feel more like two strangers who share a space at home",
    "Atr5":  "We do not have time at home as partners",
    "Atr6":  "I enjoy my holidays with my husband/wife",
    "Atr7":  "I enjoy traveling with my husband/wife",
    "Atr8":  "Most of our goals are common",
    "Atr9":  "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony",
    "Atr10": "When it comes to personal liberty, we both have similar beliefs",
    "Atr11": "We both have similar entertainment",
    "Atr12": "Most of our goals for people (children, friends, etc.) are the same",
    "Atr13": "Our dreams of living with each other are similar and harmonious",
    "Atr14": "We both are compatible with each other about what love should be",
    "Atr15": "In terms of living a good life, we both agree with each other",
    "Atr16": "Our views about the ideal marriage are similar",
    "Atr17": "We both agree on the roles that should be played in a marriage",
    "Atr18": "We both have similar values in trust",
    "Atr19": "I know exactly what my partner likes",
    "Atr20": "I know how my partner wants to be taken care of when he/she is sick",
    "Atr21": "I know my partner's favorite food",
    "Atr22": "I can tell what kind of stress my partner is facing in his/her life",
    "Atr23": "I have knowledge of my partner's inner world",
    "Atr24": "I know my partner's basic concerns",
    "Atr25": "I know what my partner's current sources of stress are",
    "Atr26": "I know my partner's hopes and wishes",
    "Atr27": "I know my husband/wife very well",
    "Atr28": "I know my partner's friends and his/her social relationships",
    "Atr29": "I feel aggressive when I argue with my husband/wife",
    "Atr30": "When discussing with my husband/wife, I usually use expressions such as 'you always' or 'you never'",
    "Atr31": "I can use negative statements about my partner's personality during our discussions",
    "Atr32": "I can use offensive expressions during our discussions",
    "Atr33": "I can insult my partner during our discussions",
    "Atr34": "I can be humiliating when we argue",
    "Atr35": "My argument with my husband/wife is not calm",
    "Atr36": "I hate my partner's way of opening a subject",
    "Atr37": "Our fights often occur suddenly",
    "Atr38": "I just start a fight with my husband/wife before I know what is going on",
    "Atr39": "When I talk to my husband/wife about something, my calm suddenly breaks",
    "Atr40": "When I argue with my husband/wife, I only go out and I do not say a word",
    "Atr41": "I am mostly stay silent to calm the environment a little bit",
    "Atr42": "Sometimes I think it is good for me to leave home for a while",
    "Atr43": "I would rather stay silent than argue with my husband/wife",
    "Atr44": "Even if I am right in the argument, I stay silent not to upset the other side",
    "Atr45": "When I argue with my husband/wife, I remain silent because I am afraid of not being able to control my anger",
    "Atr46": "I feel right in our discussions",
    "Atr47": "I have nothing to do with what I have been accused of",
    "Atr48": "I am not actually the one who is guilty about what I am accused of",
    "Atr49": "I am not the one who is wrong about problems at home",
    "Atr50": "I would not hesitate to tell my husband/wife about his/her inadequacy",
    "Atr51": "When I discuss it, I remind my husband/wife of his/her inadequacy",
    "Atr52": "I am not afraid to tell my husband/wife about his/her incompetence",
    "Atr53": "When one of us apologizes when our discussions go in a bad direction, the issue does not extend",
    "Atr54": "Even when things are challenging, I know we can put aside our disagreements",
}

# Factor groupings
PROTECTIVE = {
    "Komunikasi Sehat": ["Atr1", "Atr2", "Atr53", "Atr54"],
    "Waktu Berkualitas": ["Atr3", "Atr6", "Atr7"],
    "Kesamaan Visi & Nilai": ["Atr8","Atr9","Atr10","Atr11","Atr12","Atr13","Atr14","Atr15","Atr16","Atr17","Atr18"],
    "Love Maps (Peta Cinta)": ["Atr19","Atr20","Atr21","Atr22","Atr23","Atr24","Atr25","Atr26","Atr27","Atr28"],
}

RISK = {
    "Keterasingan": ["Atr4", "Atr5"],
    "Agresi & Kritik": ["Atr29","Atr30","Atr31","Atr32","Atr33","Atr34","Atr35","Atr36","Atr37","Atr38","Atr39","Atr50","Atr51","Atr52"],
    "Stonewalling": ["Atr40","Atr41","Atr42","Atr43","Atr44","Atr45"],
    "Defensif": ["Atr46","Atr47","Atr48","Atr49"],
}

TOP_6_CBFS = ["Atr16", "Atr15", "Atr27", "Atr20", "Atr7", "Atr3"]

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_dummy_data(n_total=200, n_divorced=75, seed=42):
    rng = np.random.default_rng(seed)
    n_married = n_total - n_divorced
    cols = list(ATTRIBUTES.keys())

    def sample_married(n):
        rows = []
        for _ in range(n):
            row = {}
            for c in cols:
                if c in [a for grp in PROTECTIVE.values() for a in grp]:
                    row[c] = rng.integers(2, 5)  # skor tinggi = protective
                elif c in ["Atr4","Atr5"]:        # keterasingan
                    row[c] = rng.integers(0, 2)
                else:                              # risk factors rendah
                    row[c] = rng.integers(0, 3)
            rows.append(row)
        return rows

    def sample_divorced(n):
        rows = []
        for _ in range(n):
            row = {}
            for c in cols:
                if c in [a for grp in PROTECTIVE.values() for a in grp]:
                    row[c] = rng.integers(0, 3)  # skor rendah = protective lemah
                elif c in [a for grp in RISK.values() for a in grp]:
                    row[c] = rng.integers(2, 5)  # risk tinggi
                else:
                    row[c] = rng.integers(1, 4)
            rows.append(row)
        return rows

    married_rows  = sample_married(n_married)
    divorced_rows = sample_divorced(n_divorced)

    df_m = pd.DataFrame(married_rows);  df_m["Label"] = 0
    df_d = pd.DataFrame(divorced_rows); df_d["Label"] = 1

    df = pd.concat([df_m, df_d], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_data
def train_models(df, use_cbfs=False):
    X = df.drop("Label", axis=1)
    y = df["Label"]

    if use_cbfs:
        X = X[TOP_6_CBFS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_sc, y_train)
    y_pred_nb = nb.predict(X_test_sc)
    results["Naive Bayes"] = {
        "model": nb, "scaler": scaler,
        "y_test": y_test, "y_pred": y_pred_nb,
        "accuracy": accuracy_score(y_test, y_pred_nb),
        "report": classification_report(y_test, y_pred_nb, target_names=["Menikah","Bercerai"], output_dict=True),
        "cm": confusion_matrix(y_test, y_pred_nb),
        "features": list(X.columns),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "model": rf, "scaler": None,
        "y_test": y_test, "y_pred": y_pred_rf,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "report": classification_report(y_test, y_pred_rf, target_names=["Menikah","Bercerai"], output_dict=True),
        "cm": confusion_matrix(y_test, y_pred_rf),
        "features": list(X.columns),
        "importances": rf.feature_importances_,
    }

    # ANN
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                        max_iter=500, random_state=42)
    ann.fit(X_train_sc, y_train)
    y_pred_ann = ann.predict(X_test_sc)
    results["ANN"] = {
        "model": ann, "scaler": scaler,
        "y_test": y_test, "y_pred": y_pred_ann,
        "accuracy": accuracy_score(y_test, y_pred_ann),
        "report": classification_report(y_test, y_pred_ann, target_names=["Menikah","Bercerai"], output_dict=True),
        "cm": confusion_matrix(y_test, y_pred_ann),
        "features": list(X.columns),
    }

    return results, X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Pengaturan")
use_cbfs = st.sidebar.checkbox("Gunakan CBFS (6 Fitur Teratas)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Keterangan Fitur CBFS**")
for f in TOP_6_CBFS:
    st.sidebar.markdown(f"- **{f}**: {ATTRIBUTES[f][:50]}...")

st.sidebar.markdown("---")
st.sidebar.markdown("**Distribusi Data Dummy**")
st.sidebar.markdown("- Total: 200 partisipan")
st.sidebar.markdown("- Menikah: 125")
st.sidebar.markdown("- Bercerai: 75")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.title("Divorce Prediction Pipeline")
st.markdown("Implementasi pipeline machine learning berbasis Divorce Predictor Scale (DPS) dan Gottman Couples Therapy, menggunakan data dummy 200 partisipan.")

df = generate_dummy_data()
results, X_train, X_test, y_train, y_test = train_models(df, use_cbfs=use_cbfs)

# ─── TAB NAVIGATION ───────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset & EDA",
    "Penjelasan Algoritma",
    "Hasil Model",
    "Perbandingan",
    "Prediksi Individu",
])

# ══════════════════════════════════════════════
# TAB 1 — DATASET & EDA
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Dummy")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Partisipan", 200)
    col2.metric("Menikah", 125)
    col3.metric("Bercerai", 75)
    col4.metric("Jumlah Fitur", 54 if not use_cbfs else 6)

    st.markdown("---")

    # Sample data
    st.markdown("**Sampel Data (10 baris pertama)**")
    display_df = df.copy()
    display_df["Label"] = display_df["Label"].map({0: "Menikah", 1: "Bercerai"})
    st.dataframe(display_df.head(10), use_container_width=True)

    st.markdown("---")

    # Distribusi label
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distribusi Kelas**")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df["Label"].value_counts()
        bars = ax.bar(["Menikah", "Bercerai"], [counts[0], counts[1]],
                      color=["#4C72B0", "#DD8452"])
        ax.bar_label(bars, padding=3)
        ax.set_ylabel("Jumlah")
        ax.set_ylim(0, 160)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Rata-rata Skor per Kelompok Faktor**")
        protective_cols = [a for grp in PROTECTIVE.values() for a in grp]
        risk_cols       = [a for grp in RISK.values() for a in grp]

        married_df  = df[df["Label"] == 0]
        divorced_df = df[df["Label"] == 1]

        categories = list(PROTECTIVE.keys()) + list(RISK.keys())
        all_groups = {**PROTECTIVE, **RISK}
        m_means = [married_df[cols].mean().mean() for cols in all_groups.values()]
        d_means = [divorced_df[cols].mean().mean() for cols in all_groups.values()]

        x = np.arange(len(categories))
        w = 0.35
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.bar(x - w/2, m_means, w, label="Menikah",  color="#4C72B0")
        ax2.bar(x + w/2, d_means, w, label="Bercerai", color="#DD8452")
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=25, ha="right", fontsize=8)
        ax2.set_ylabel("Rata-rata Skor")
        ax2.legend(fontsize=8)
        ax2.spines[["top","right"]].set_visible(False)
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown("**Pemetaan 54 Atribut ke Risk & Protective Factors**")

    col_p, col_r = st.columns(2)
    with col_p:
        st.markdown("Protective Factors")
        for group, atrs in PROTECTIVE.items():
            with st.expander(f"{group} ({len(atrs)} item)"):
                for a in atrs:
                    st.markdown(f"- **{a}**: {ATTRIBUTES[a]}")

    with col_r:
        st.markdown("Risk Factors")
        for group, atrs in RISK.items():
            with st.expander(f"{group} ({len(atrs)} item)"):
                for a in atrs:
                    st.markdown(f"- **{a}**: {ATTRIBUTES[a]}")

# ══════════════════════════════════════════════
# TAB 2 — PENJELASAN ALGORITMA
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Penjelasan Tiga Algoritma Machine Learning")

    # Pipeline diagram (teks)
    st.markdown("""
    **Alur Pipeline Secara Umum**

    ```
    DATA MENTAH (200 baris, 54 fitur)
         |
         v
    [1. Preprocessing]  --> Isi missing value, normalisasi (StandardScaler)
         |
         v
    [2. Feature Selection (opsional)]  --> CBFS: pilih 6 fitur terkorelasi tertinggi
         |
         v
    [3. Data Splitting]  --> 60% Training / 40% Testing
         |
         v
    [4. Training Algoritma]  --> NB  |  Random Forest  |  ANN
         |
         v
    [5. Evaluasi]  --> Accuracy, Kappa, Confusion Matrix, Classification Report
    ```
    """)

    st.markdown("---")

    # NB
    with st.expander("Naive Bayes (NB) - Probabilistik Classifier", expanded=True):
        st.markdown("""
**Apa itu Naive Bayes?**

Naive Bayes adalah algoritma klasifikasi berbasis probabilitas yang menerapkan Teorema Bayes.
Disebut "naive" (naif) karena algoritma ini mengasumsikan bahwa setiap fitur bersifat **independen satu sama lain** — sebuah asumsi yang disederhanakan, namun terbukti efektif di banyak kasus nyata.

**Cara kerja singkat:**

Algoritma menghitung peluang bahwa seseorang termasuk kelas "Bercerai" atau "Menikah" berdasarkan skor tiap atribut DPS.
Kelas dengan probabilitas tertinggi yang menjadi prediksi akhir.

**Formula Bayes:**

    P(Kelas | Data) = P(Data | Kelas) x P(Kelas) / P(Data)

**Kelebihan:**
- Sangat cepat dan ringan secara komputasi
- Bekerja baik pada dataset kecil
- Mudah diinterpretasikan

**Kekurangan:**
- Asumsi independensi antar fitur jarang terpenuhi di data nyata
- Performa bisa menurun jika fitur saling berkorelasi kuat

**Hyperparameter dalam studi ini:**
- var_smoothing: 1e-9 (mencegah probabilitas nol)
        """)

    # RF
    with st.expander("Random Forest (RF) - Ensemble Learning", expanded=True):
        st.markdown("""
**Apa itu Random Forest?**

Random Forest adalah algoritma ensemble yang membangun **banyak pohon keputusan (decision trees)** secara paralel, 
lalu menggabungkan hasil prediksi mereka melalui mekanisme **voting mayoritas**.

Kata "Random" merujuk pada dua sumber keacakan yang membuat setiap pohon berbeda:
1. Setiap pohon dilatih pada subset data (bootstrap sampling)
2. Setiap pembelahan node hanya mempertimbangkan sebagian fitur secara acak

**Cara kerja singkat:**

    Data --> [Pohon 1] --> Prediksi A
         --> [Pohon 2] --> Prediksi B    --> Voting Mayoritas --> Hasil Akhir
         --> [Pohon N] --> Prediksi C

**Kelebihan:**
- Akurasi tinggi dan robust terhadap overfitting
- Mampu menangani fitur yang tidak seimbang dan data yang hilang
- Menghasilkan feature importance secara otomatis
- Tidak memerlukan normalisasi data

**Kekurangan:**
- Model lebih sulit diinterpretasikan dibanding pohon tunggal
- Lebih lambat dari NB untuk dataset besar

**Hyperparameter dalam studi ini:**
- n_estimators = 100 (jumlah pohon)
- max_depth = 5 (kedalaman maksimum pohon)
- random_state = 42
        """)

    # ANN
    with st.expander("Artificial Neural Network (ANN) - Deep Learning Ringan", expanded=True):
        st.markdown("""
**Apa itu ANN?**

ANN adalah algoritma yang terinspirasi dari cara kerja otak manusia. Terdiri dari lapisan-lapisan neuron buatan 
yang saling terhubung dan belajar dari data melalui proses penyesuaian bobot (weights).

**Arsitektur dalam studi ini:**

    INPUT LAYER       HIDDEN LAYER 1    HIDDEN LAYER 2    OUTPUT LAYER
    (54 fitur)   -->  (64 neuron, ReLU) --> (32 neuron, ReLU) --> (2 kelas)
                       Menikah / Bercerai

**Cara kerja singkat:**
1. Data masuk ke input layer
2. Setiap neuron menghitung nilai berbobot dari input
3. Fungsi aktivasi ReLU memutuskan apakah neuron "aktif" atau tidak
4. Proses backpropagation menyesuaikan bobot berdasarkan kesalahan prediksi
5. Diulang selama N epoch hingga model konvergen

**Kelebihan:**
- Mampu menangkap pola non-linear yang kompleks
- Sangat fleksibel dan dapat diskalakan

**Kekurangan:**
- Membutuhkan lebih banyak data dan waktu pelatihan
- Sulit diinterpretasikan (black box)
- Sensitif terhadap normalisasi data dan hyperparameter

**Hyperparameter dalam studi ini:**
- hidden_layer_sizes = (64, 32)
- activation = 'relu'
- max_iter = 500
        """)

    st.markdown("---")
    st.markdown("**Peran CBFS (Correlation-Based Feature Selection)**")
    st.markdown("""
CBFS memilih subset fitur yang paling berkorelasi dengan label kelas (Menikah/Bercerai) 
tetapi tidak saling berkorelasi satu sama lain. Hasilnya adalah 6 fitur terbaik dari 54 atribut DPS.

| Fitur | Nilai Signifikansi | Deskripsi |
|-------|-------------------|-----------|
| Atr16 | 0.601 | Kesamaan pandangan soal pernikahan ideal |
| Atr15 | 0.589 | Kesamaan pandangan soal hidup yang baik |
| Atr27 | 0.584 | Mengenal pasangan dengan baik |
| Atr20 | 0.584 | Mengetahui cara merawat pasangan saat sakit |
| Atr7  | 0.571 | Menikmati perjalanan bersama pasangan |
| Atr3  | 0.570 | Waktu bersama pasangan terasa istimewa |

Seluruh 6 fitur teratas berasal dari kelompok **Protective Factors** — 
mengindikasikan bahwa lemahnya fondasi positif lebih prediktif terhadap perceraian daripada tingginya faktor risiko.
    """)

# ══════════════════════════════════════════════
# TAB 3 — HASIL MODEL
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Hasil Evaluasi per Algoritma")
    mode_label = "dengan CBFS (6 Fitur)" if use_cbfs else "dengan 54 Fitur (tanpa CBFS)"
    st.markdown(f"Mode aktif: **{mode_label}**")

    for model_name, res in results.items():
        with st.expander(f"{model_name} — Akurasi: {res['accuracy']*100:.2f}%", expanded=True):
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown("**Confusion Matrix**")
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = res["cm"]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Menikah","Bercerai"],
                            yticklabels=["Menikah","Bercerai"])
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")
                ax.set_title(model_name)
                st.pyplot(fig)
                plt.close()

            with col_b:
                st.markdown("**Classification Report**")
                report = res["report"]
                report_df = pd.DataFrame({
                    "Precision": [report["Menikah"]["precision"], report["Bercerai"]["precision"]],
                    "Recall":    [report["Menikah"]["recall"],    report["Bercerai"]["recall"]],
                    "F1-Score":  [report["Menikah"]["f1-score"],  report["Bercerai"]["f1-score"]],
                    "Support":   [int(report["Menikah"]["support"]), int(report["Bercerai"]["support"])],
                }, index=["Menikah", "Bercerai"])
                st.dataframe(report_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}))

                st.markdown(f"""
| Metrik | Nilai |
|--------|-------|
| Accuracy | {res['accuracy']*100:.2f}% |
| Macro F1 | {report['macro avg']['f1-score']*100:.2f}% |
                """)

            # Feature importance khusus RF
            if model_name == "Random Forest" and not use_cbfs:
                st.markdown("**Feature Importance (Top 15)**")
                feat_imp = pd.Series(res["importances"], index=res["features"]).sort_values(ascending=False)
                top15 = feat_imp.head(15)

                # Beri warna berdasarkan kelompok
                protect_list = [a for grp in PROTECTIVE.values() for a in grp]
                risk_list    = [a for grp in RISK.values() for a in grp]
                colors = []
                for f in top15.index:
                    if f in protect_list:
                        colors.append("#4C72B0")
                    elif f in risk_list:
                        colors.append("#DD8452")
                    else:
                        colors.append("gray")

                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
                ax3.set_xlabel("Importance")
                ax3.set_title("Feature Importance - Random Forest")
                ax3.spines[["top","right"]].set_visible(False)
                patch_p = mpatches.Patch(color="#4C72B0", label="Protective Factor")
                patch_r = mpatches.Patch(color="#DD8452", label="Risk Factor")
                ax3.legend(handles=[patch_p, patch_r], fontsize=8)
                st.pyplot(fig3)
                plt.close()

# ══════════════════════════════════════════════
# TAB 4 — PERBANDINGAN
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Perbandingan Kinerja Semua Algoritma")

    # Summary table
    summary = []
    for name, res in results.items():
        r = res["report"]
        summary.append({
            "Algoritma": name,
            "Accuracy (%)": round(res["accuracy"]*100, 2),
            "Precision (macro)": round(r["macro avg"]["precision"]*100, 2),
            "Recall (macro)": round(r["macro avg"]["recall"]*100, 2),
            "F1-Score (macro)": round(r["macro avg"]["f1-score"]*100, 2),
        })

    summary_df = pd.DataFrame(summary)
    best_idx = summary_df["Accuracy (%)"].idxmax()
    st.dataframe(summary_df.style.highlight_max(subset=["Accuracy (%)"], color="#c6efce"), use_container_width=True)
    st.caption(f"Algoritma terbaik: **{summary_df.loc[best_idx, 'Algoritma']}** dengan akurasi {summary_df.loc[best_idx, 'Accuracy (%)']:.2f}%")

    st.markdown("---")

    # Bar chart perbandingan
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary_df))
    w = 0.2
    metrics = ["Accuracy (%)", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"]
    clrs = ["#4C72B0","#55A868","#C44E52","#8172B2"]
    for i, (m, c) in enumerate(zip(metrics, clrs)):
        ax4.bar(x + i*w, summary_df[m], w, label=m, color=c)
    ax4.set_xticks(x + w*1.5)
    ax4.set_xticklabels(summary_df["Algoritma"])
    ax4.set_ylim(50, 105)
    ax4.set_ylabel("Nilai (%)")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.spines[["top","right"]].set_visible(False)
    st.pyplot(fig4)
    plt.close()

    st.markdown("---")
    st.markdown("**Catatan Interpretasi**")
    st.markdown("""
- **Accuracy**: Persentase total prediksi yang benar dari seluruh data uji.
- **Precision**: Dari semua yang diprediksi "Bercerai", berapa persen yang benar-benar bercerai.
- **Recall**: Dari semua yang benar-benar bercerai, berapa persen yang berhasil terdeteksi.
- **F1-Score**: Rata-rata harmonik precision dan recall — lebih representatif saat data tidak seimbang.

Dalam konteks konseling pernikahan, **recall untuk kelas "Bercerai"** adalah metrik yang paling kritis, 
karena kita tidak ingin melewatkan pasangan yang berisiko tinggi.
    """)

    # CBFS vs non-CBFS note
    st.markdown("---")
    st.info("Aktifkan opsi 'Gunakan CBFS' di sidebar untuk melihat perubahan performa saat hanya menggunakan 6 fitur teratas. Sesuai penelitian Moumen et al. (2024), RF dengan CBFS mencapai akurasi tertinggi 91.66%.")

# ══════════════════════════════════════════════
# TAB 5 — PREDIKSI INDIVIDU
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Simulasi Prediksi untuk Satu Pasangan")
    st.markdown("Masukkan skor untuk setiap pernyataan DPS (0 = Tidak pernah, 4 = Selalu).")

    st.markdown("---")
    st.markdown("**Cara cepat: pilih profil preset**")
    preset = st.selectbox("Pilih profil", [
        "-- Isi manual --",
        "Pasangan Harmonis (Protective tinggi, Risk rendah)",
        "Pasangan Berisiko (Protective rendah, Risk tinggi)",
    ])

    protective_all = [a for grp in PROTECTIVE.values() for a in grp]
    risk_all       = [a for grp in RISK.values() for a in grp]

    input_vals = {}

    if preset == "Pasangan Harmonis (Protective tinggi, Risk rendah)":
        for a in ATTRIBUTES:
            if a in protective_all:
                input_vals[a] = 4
            else:
                input_vals[a] = 1
    elif preset == "Pasangan Berisiko (Protective rendah, Risk tinggi)":
        for a in ATTRIBUTES:
            if a in risk_all:
                input_vals[a] = 4
            elif a in ["Atr4","Atr5"]:
                input_vals[a] = 4
            else:
                input_vals[a] = 1
    else:
        for a in ATTRIBUTES:
            input_vals[a] = 2

    st.markdown("---")
    st.markdown("**Input Skor per Kategori**")

    all_groups = {**PROTECTIVE, **RISK}
    for group_name, atrs in all_groups.items():
        is_risk = group_name in RISK
        group_label = f"[RISK] {group_name}" if is_risk else f"[Protective] {group_name}"
        with st.expander(group_label):
            cols = st.columns(2)
            for i, a in enumerate(atrs):
                with cols[i % 2]:
                    input_vals[a] = st.slider(
                        f"{a}: {ATTRIBUTES[a][:60]}...",
                        min_value=0, max_value=4,
                        value=input_vals[a], key=f"slider_{a}"
                    )

    if st.button("Jalankan Prediksi", type="primary"):
        st.markdown("---")
        input_row = pd.DataFrame([input_vals])

        prot_score = input_row[protective_all].mean(axis=1).values[0]
        risk_score = input_row[risk_all].mean(axis=1).values[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata Protective Score", f"{prot_score:.2f} / 4")
        col2.metric("Rata-rata Risk Score", f"{risk_score:.2f} / 4")
        col3.metric("Selisih (Protective - Risk)", f"{prot_score - risk_score:+.2f}")

        st.markdown("---")
        st.markdown("**Prediksi dari Setiap Algoritma**")

        pred_cols = st.columns(3)
        for idx, (model_name, res) in enumerate(results.items()):
            feats = res["features"]
            model = res["model"]
            scaler = res["scaler"]

            X_input = input_row[feats]
            if scaler is not None:
                X_input_sc = scaler.transform(X_input)
            else:
                X_input_sc = X_input

            pred = model.predict(X_input_sc)[0]
            proba = model.predict_proba(X_input_sc)[0]

            label = "Bercerai" if pred == 1 else "Menikah"
            conf  = proba[pred] * 100

            with pred_cols[idx]:
                st.markdown(f"**{model_name}**")
                if pred == 1:
                    st.error(f"Prediksi: {label}")
                else:
                    st.success(f"Prediksi: {label}")
                st.markdown(f"Confidence: **{conf:.1f}%**")
                st.markdown(f"P(Menikah): {proba[0]*100:.1f}%")
                st.markdown(f"P(Bercerai): {proba[1]*100:.1f}%")

        # Radar chart per kategori
        st.markdown("---")
        st.markdown("**Profil Skor per Sub-Kategori (Radar Chart)**")

        cat_names = list(all_groups.keys())
        cat_scores = []
        for group_name, atrs in all_groups.items():
            cat_scores.append(input_row[atrs].mean(axis=1).values[0])

        angles = np.linspace(0, 2*np.pi, len(cat_names), endpoint=False).tolist()
        cat_scores_plot = cat_scores + [cat_scores[0]]
        angles += angles[:1]

        fig5, ax5 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax5.plot(angles, cat_scores_plot, "o-", linewidth=2, color="#4C72B0")
        ax5.fill(angles, cat_scores_plot, alpha=0.25, color="#4C72B0")
        ax5.set_thetagrids(np.degrees(angles[:-1]), cat_names, fontsize=8)
        ax5.set_ylim(0, 4)
        ax5.set_title("Profil Pasangan per Sub-Kategori", pad=20)
        ax5.axhline(y=2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        st.pyplot(fig5)
        plt.close()

        st.markdown("""
*Garis putus-putus di tengah (nilai 2) adalah batas tengah skala.*
- Sub-kategori Protective di atas 2 menunjukkan kekuatan hubungan.
- Sub-kategori Risk di atas 2 menunjukkan area yang perlu perhatian terapis.
        """)
