
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Dashboard Avançado - Diabetes", layout="wide")

st.title("🧠 Dashboard Avançado de Análise de Diabetes")
st.markdown("Este painel apresenta gráficos dos algoritmos de machine learning aplicados à base de diabetes.")

# Upload do CSV
uploaded_file = st.file_uploader("Faça upload do arquivo diabetes.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.header("📊 Visão Geral dos Dados")
    st.dataframe(df.describe())

    # Tratamento de dados
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_cleaned = df.copy()
    for col in cols_to_fix:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].replace(0, median_val)

    # Features e alvo
    X = df_cleaned.drop(columns=["Outcome"])
    y = df_cleaned["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Modelos
    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    results = {}
    conf_matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = np.round((preds == y_test).mean(), 4)
        results[name] = acc
        conf_matrices[name] = confusion_matrix(y_test, preds)

    tab1, tab2, tab3 = st.tabs(["📈 Acurácia dos Modelos", "🔗 Matriz de Correlação", "🧩 Matrizes de Confusão"])

    with tab1:
        st.subheader("Comparação de Acurácia entre os Modelos")
        acc_df = pd.DataFrame.from_dict(results, orient="index", columns=["Acurácia"]).sort_values(by="Acurácia", ascending=False)
        st.bar_chart(acc_df)

    with tab2:
        st.subheader("Matriz de Correlação das Variáveis")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_cleaned.corr(), annot=True, cmap="coolwarm", square=True, ax=ax_corr)
        st.pyplot(fig_corr)

    with tab3:
        st.subheader("Matrizes de Confusão por Modelo")
        for name, matrix in conf_matrices.items():
            st.markdown(f"**{name}**")
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Não Diabético", "Diabético"])
            disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
            plt.grid(False)
            st.pyplot(fig_cm)

    st.markdown("---")
    st.caption("Feito com ❤️ usando Streamlit.")

else:
    st.info("Por favor, carregue o arquivo diabetes.csv para visualizar o dashboard.")
