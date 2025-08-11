import io

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Dados e Modelagem",
    page_icon="📊",
    layout="wide"
)

# --- Título e Descrição ---
st.title("📊 Ferramenta Completa de Análise e Modelagem de Machine Learning")
st.write(
    "Esta ferramenta interativa combina uma Análise Exploratória de Dados (EDA) detalhada "
    "com um poderoso Simulador de Modelos de Machine Learning."
)
st.write("---")

# --- Barra Lateral ---
with st.sidebar:
    st.header("1. Carregue seus Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    params = {}

# --- Lógica Principal ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        st.error("Erro: Após a limpeza, o conjunto de dados ficou vazio.")
        st.stop()

    st.success("Arquivo carregado e dados limpos com sucesso!")

    # --- Divisão em Abas Principais ---
    eda_tab, model_tab = st.tabs(["Análise Exploratória (EDA)", "Simulador de Modelos"])

    # =====================================================================================
    # --- ABA 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---
    # =====================================================================================
    with eda_tab:
        st.header("Análise Exploratória dos Dados")
        st.subheader("1. Visão Geral do Dataset")
        st.dataframe(df.head())
        st.subheader("2. Estatísticas Descritivas")
        st.dataframe(df.describe())
        st.subheader("3. Distribuição das Variáveis")
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_columns:
            st.warning("Não há colunas numéricas para plotar a distribuição.")
        else:
            selected_col = st.selectbox("Selecione uma variável para ver sua distribuição:", options=numeric_columns)
            if selected_col:
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.histplot(df[selected_col], kde=True, ax=ax)
                ax.set_title(f'Distribuição de {selected_col}')
                st.pyplot(fig)

        st.subheader("4. Matriz de Correlação entre Variáveis")
        if not numeric_columns:
            st.warning("Não há colunas numéricas para calcular a correlação.")
        else:
            corr_matrix = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))

            # MUDANÇA APLICADA AQUI para melhorar a legibilidade
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax,
                annot_kws={"size": 6}  # Controla o tamanho da fonte dos números
            )
            ax.set_title("Mapa de Calor da Correlação")
            st.pyplot(fig)

    # =====================================================================================
    # --- ABA 2: SIMULADOR DE MODELOS ---
    # =====================================================================================
    with model_tab:
        st.header("Simulador de Modelos de Machine Learning")
        with st.sidebar:
            st.header("2. Configuração do Modelo")
            model_type = st.selectbox("Selecione o Modelo",
                                      ("Regressão Logística", "Árvore de Decisão", "Random Forest"))
            if model_type == "Regressão Logística":
                st.subheader("Hiperparâmetros")
                params['C'] = st.slider("Regularização (C)", 0.01, 10.0, 1.0)
                params['max_iter'] = st.slider("Iterações", 100, 5000, 1000)
            elif model_type == "Árvore de Decisão":
                st.subheader("Hiperparâmetros")
                params['max_depth'] = st.slider("Profundidade Máxima", 2, 30, 10)
                params['criterion'] = st.selectbox("Critério", ("gini", "entropy"))
            elif model_type == "Random Forest":
                st.subheader("Hiperparâmetros")
                params['n_estimators'] = st.slider("Nº de Árvores", 10, 500, 100)
                params['max_depth'] = st.slider("Profundidade Máxima", 2, 30, 10, key="rf_depth")
                params['criterion'] = st.selectbox("Critério", ("gini", "entropy"), key="rf_criterion")

        colunas = df.columns.tolist()
        target_variable = st.selectbox("Selecione a sua variável-alvo:", options=colunas, key="target_model")
        features_disponiveis = [col for col in colunas if col != target_variable]
        feature_variables = st.multiselect("Selecione as suas features:", options=features_disponiveis,
                                           default=features_disponiveis)
        st.write("---")

        if st.button("Treinar Modelo"):
            with st.spinner("Treinando e avaliando o modelo..."):
                X = df[feature_variables]
                y = df[target_variable]
                if y.dtype == 'object' or y.dtype.name == 'category':
                    unique_vals = y.unique()
                    if len(unique_vals) == 2:
                        y = y.map({unique_vals[0]: 0, unique_vals[1]: 1})
                    else:
                        st.error("Variável-alvo precisa ter 2 classes para a Curva ROC."); st.stop()
                X = pd.get_dummies(X, drop_first=True)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

                if model_type == "Regressão Logística":
                    model = LogisticRegression(**params)
                elif model_type == "Árvore de Decisão":
                    model = DecisionTreeClassifier(**params)
                else:
                    model = RandomForestClassifier(**params)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)

                st.success("Modelo treinado com sucesso!")
                st.subheader("Resultados da Avaliação do Modelo")

                res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
                    ["Métricas", "Matriz de Confusão", "Comparativo & Curva ROC", "Probabilidades"])

                with res_tab1:
                    st.metric(label="Acurácia", value=f"{accuracy:.2%}")
                    st.dataframe(pd.DataFrame(
                        classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose())

                with res_tab2:
                    st.subheader("Matriz de Confusão")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3.75), dpi=80)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Previsto')
                    ax.set_ylabel('Verdadeiro')
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf, width=550)  # aqui controla a largura da imagem em px

                with res_tab3:
                    st.subheader("Curva ROC")
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(5, 3.75), dpi=80)
                    ax.plot(fpr, tpr, color='blue', label=f'Curva ROC (área = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    ax.set_xlabel('Taxa de Falsos Positivos')
                    ax.set_ylabel('Taxa de Verdadeiros Positivos')
                    ax.legend(loc="lower right")
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf, width=500)  # controle da largura em px

                    st.subheader("Comparativo de Modelos (Acurácia)")
                    resultados = {};
                    for nome, modelo in {"Regressão Logística": LogisticRegression(max_iter=1000),
                                         "Árvore de Decisão": DecisionTreeClassifier(),
                                         "Random Forest": RandomForestClassifier()}.items():
                        modelo.fit(X_train, y_train);
                        pred = modelo.predict(X_test);
                        resultados[nome] = accuracy_score(y_test, pred)
                    st.dataframe(pd.DataFrame.from_dict(resultados, orient='index', columns=['Acurácia']))

                with res_tab4:
                    st.subheader("Probabilidade por Amostra")
                    st.dataframe(
                        pd.DataFrame({"Real": y_test.values, "Previsto": y_pred, "Prob. Classe 1": y_prob}).sort_values(
                            "Prob. Classe 1", ascending=False))
else:
    st.warning("Por favor, carregar um arquivo CSV para começar.")