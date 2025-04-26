
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

st.set_page_config(page_title="Dashboard Ultra Pro - Diabetes", layout="wide")

st.title("🧠 Dashboard Avançado de Análise de Diabetes")
st.markdown("Análise completa de dados, modelos de machine learning e insights estratégicos sobre a base de Diabetes do Kaggle.")

uploaded_file = st.file_uploader("Faça upload do arquivo diabetes.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("Configurações")
    model_select = st.sidebar.selectbox("Escolha o Modelo para Análises:", ["Regressão Logística", "Random Forest", "KNN", "SVM"])

    st.header("📊 Visão Geral dos Dados")
    st.dataframe(df.describe())

    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_cleaned = df.copy()
    for col in cols_to_fix:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].replace(0, median_val)

    X = df_cleaned.drop(columns=["Outcome"])
    y = df_cleaned["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    results = {}
    conf_matrices = {}
    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = np.round((preds == y_test).mean(), 4)
        results[name] = acc
        conf_matrices[name] = confusion_matrix(y_test, preds)
        reports[name] = classification_report(y_test, preds, output_dict=True)

    # Random Forest para Feature Importance
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    feature_importance = pd.DataFrame({
        'Feature': df_cleaned.drop(columns=['Outcome']).columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Acurácia dos Modelos",
        "🔗 Matriz de Correlação",
        "🧩 Matrizes de Confusão",
        "💡 Insights e Conclusões",
        "🏆 Importância das Variáveis",
        "📋 Métricas do Modelo"
    ])

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

    with tab4:
        st.subheader("Principais Insights Estratégicos")
        insights = {
            "🔹 Glicose alta é o principal indicador de risco.": "Altos níveis de glicose no plasma estão fortemente associados ao diagnóstico positivo de diabetes.",
            "🔹 IMC elevado é um fator importante.": "Pacientes com BMI acima de 30 têm maior risco de desenvolver diabetes.",
            "🔹 Idade impacta o risco.": "O risco de diabetes aumenta significativamente a partir dos 40 anos.",
            "🔹 Histórico familiar influencia.": "A função de pedigree para diabetes também impacta a chance de diagnóstico, mas com menor peso."
        }
        for k, v in insights.items():
            st.success(f"{k}\n\n{v}")


    with tab5:
        st.subheader("Importância das Variáveis")
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis", ax=ax_imp)
        st.pyplot(fig_imp)

    with tab6:
        st.subheader(f"Métricas do Modelo: {model_select}")
        selected_report = reports[model_select]
        col1, col2, col3 = st.columns(3)
        col1.metric("Acurácia", f"{selected_report['accuracy']*100:.2f}%")
        col2.metric("Recall (Detecção)", f"{selected_report['1']['recall']*100:.2f}%")
        col3.metric("Precisão", f"{selected_report['1']['precision']*100:.2f}%")

    st.markdown("---")
    st.caption("Feito com ❤️ usando Streamlit.")

else:
    st.info("Por favor, carregue o arquivo diabetes.csv para visualizar o dashboard.")
