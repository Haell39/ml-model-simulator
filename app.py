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

st.set_page_config(page_title="Análise de Dados e Modelagem", page_icon="📊", layout="wide")

st.title("📊 Ferramenta de Análise de Dados e Machine Learning")
st.write("Uma solução para análise de dados e modelagem preditiva. Esta aplicação integra um módulo de Análise Exploratória (EDA) com histogramas, mapas de calor e gráficos de dispersão, a um poderoso simulador que permite treinar, ajustar e avaliar múltiplos algoritmos de Machine Learning, incluindo a visualização de Curvas ROC e métricas completas.")

with st.sidebar:
    st.header("1. Carregue seus Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    params = {}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        st.error("Erro: Após a limpeza, o conjunto de dados ficou vazio.")
        st.stop()

    st.success("Arquivo carregado e dados limpos com sucesso!")
    eda_tab, model_tab = st.tabs(["Análise Exploratória (EDA)", "Simulador de Modelos"])

    with eda_tab:
        st.header("Análise Exploratória dos Dados")

        # 1. Visão Geral
        st.subheader("1. Visão Geral do Dataset")
        st.dataframe(df.head())

        # 2. Estatísticas
        st.subheader("2. Estatísticas Descritivas")
        st.dataframe(df.describe())

        # 3. Distribuição
        st.subheader("3. Distribuição das Variáveis Numéricas")
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            selected_col = st.selectbox("Selecione uma variável:", options=numeric_columns)
            if selected_col:
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.histplot(df[selected_col], kde=True, ax=ax)
                ax.set_title(f'Distribuição de {selected_col}')
                st.pyplot(fig)
        else:
            st.warning("Não há colunas numéricas para plotar.")

        # 4. Correlação
        st.subheader("4. Matriz de Correlação")
        if numeric_columns:
            corr_matrix = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 6})
            ax.set_title("Mapa de Calor da Correlação")
            st.pyplot(fig)
        else:
            st.warning("Não há colunas numéricas para calcular correlação.")

        # 5. Variáveis Categóricas
        st.subheader("5. Análise de Variáveis Categóricas")
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns:
            cat_col = st.selectbox("Selecione uma variável categórica para análise:", options=categorical_columns,
                                   key='cat_eda')
            if cat_col:
                st.write("Contagem de valores:")
                st.dataframe(df[cat_col].value_counts())
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.countplot(x=df[cat_col], ax=ax, order=df[cat_col].value_counts().index)
                ax.set_title(f'Contagem de {cat_col}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.info("O dataset não possui variáveis categóricas para análise.")

        # 6. Scatter Plot
        st.subheader("6. Relação entre Duas Variáveis (Scatter Plot)")
        if len(numeric_columns) >= 2:
            col1 = st.selectbox("Selecione a variável para o eixo X:", options=numeric_columns, index=0)
            col2 = st.selectbox("Selecione a variável para o eixo Y:", options=numeric_columns, index=1)
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
            ax.set_title(f'Relação entre {col1} e {col2}')
            st.pyplot(fig)
        else:
            st.info("São necessárias pelo menos duas colunas numéricas para criar um gráfico de dispersão.")

        # 7. Box Plot
        st.subheader("7. Análise de Outliers (Box Plot)")
        if numeric_columns:
            outlier_col = st.selectbox("Selecione uma variável para análise de outliers:", options=numeric_columns,
                                       key='outlier_eda')
            if outlier_col:
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.boxplot(x=df[outlier_col], ax=ax)
                ax.set_title(f'Box Plot de {outlier_col}')
                st.pyplot(fig)
        else:
            st.info("Não há colunas numéricas para a análise de outliers.")

    with model_tab:
        st.header("Simulador de Modelos de Machine Learning")
        with st.sidebar:
            st.header("2. Configuração do Modelo")
            model_type = st.selectbox("Selecione o Modelo", ("Regressão Logística", "Árvore de Decisão", "Random Forest"))
            if model_type == "Regressão Logística":
                params['C'] = st.slider("Regularização (C)", 0.01, 10.0, 1.0)
                params['max_iter'] = st.slider("Iterações", 100, 5000, 1000)
            elif model_type == "Árvore de Decisão":
                params['max_depth'] = st.slider("Profundidade Máxima", 2, 30, 10)
                params['criterion'] = st.selectbox("Critério", ("gini", "entropy"))
            elif model_type == "Random Forest":
                params['n_estimators'] = st.slider("Nº de Árvores", 10, 500, 100)
                params['max_depth'] = st.slider("Profundidade Máxima", 2, 30, 10, key="rf_depth")
                params['criterion'] = st.selectbox("Critério", ("gini", "entropy"), key="rf_criterion")

        colunas = df.columns.tolist()
        target_variable = st.selectbox("Selecione a variável-alvo:", options=colunas, key="target_model")
        features_disponiveis = [col for col in colunas if col != target_variable]
        feature_variables = st.multiselect("Selecione as features:", options=features_disponiveis, default=features_disponiveis)
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
                        st.error("Variável-alvo precisa ter 2 classes para a Curva ROC.")
                        st.stop()
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
                    ["Métricas", "Matriz de Confusão", "Comparativo & Curva ROC", "Probabilidades"]
                )

                with res_tab1:
                    st.metric(label="Acurácia", value=f"{accuracy:.2%}")
                    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose())

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
                    st.image(buf, width=550)

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
                    st.image(buf, width=500)

                    st.subheader("Comparativo de Modelos (Acurácia)")
                    resultados = {}
                    for nome, modelo in {
                        "Regressão Logística": LogisticRegression(max_iter=1000),
                        "Árvore de Decisão": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier()
                    }.items():
                        modelo.fit(X_train, y_train)
                        pred = modelo.predict(X_test)
                        resultados[nome] = accuracy_score(y_test, pred)

                    st.markdown(
                        """
                        <style>
                        .df-container {max-width: 400px; margin-left: auto; margin-right: auto;}
                        </style>
                        """, unsafe_allow_html=True
                    )
                    st.markdown('<div class="df-container">', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame.from_dict(resultados, orient='index', columns=['Acurácia']))
                    st.markdown('</div>', unsafe_allow_html=True)

                with res_tab4:
                    st.subheader("Probabilidade por Amostra")
                    st.dataframe(pd.DataFrame({
                        "Real": y_test.values,
                        "Previsto": y_pred,
                        "Prob. Classe 1": y_prob
                    }).sort_values("Prob. Classe 1", ascending=False))
else:
    st.markdown("---")
    st.header("Bem-vindo ao Insight Navigator! 🚀")

    st.markdown("""
        Esta é a sua central de controle para projetos de Ciência de Dados. A ferramenta foi projetada para guiá-lo através do ciclo completo: da exploração inicial de um dataset até a avaliação detalhada de modelos preditivos.
        """)

    st.info("👈 **Para começar, carregue um arquivo CSV usando o menu na barra lateral esquerda.**")

    st.subheader("Como Utilizar a Ferramenta:")
    st.markdown("""
        1.  **Carregue seus Dados:** No menu lateral, clique em 'Browse files' e selecione um arquivo CSV do seu computador.
        2.  **Explore (Aba EDA):** Após o upload, a primeira aba lhe dará uma análise completa do seu dataset. Navegue pelas estatísticas, distribuições e correlações para entender seus dados a fundo.
        3.  **Modele (Aba Simulador):**
            * Na segunda aba, selecione sua **variável-alvo** e as **features**.
            * Na barra lateral, escolha um modelo de Machine Learning e ajuste seus **hiperparâmetros**.
            * Clique em **'Treinar Modelo'** e explore os resultados nas diversas abas de avaliação.
        """)

    st.subheader("Não tem um dataset? Sem problemas!")
    st.markdown("Você pode baixar um dataset clássico de exemplo sobre diagnóstico de câncer de mama para testar todas as funcionalidades da ferramenta.")

    with open("data/Breast Cancer Wisconsin.csv", "r") as file:
        csv_data = file.read()

    #  botão de download
    st.download_button(
        label="Clique aqui para baixar o dataset de exemplo",
        data=csv_data,
        file_name='Breast Cancer Wisconsin.csv',  # Nome do arquivo para o usuário
        mime='text/csv',
    )

    st.markdown("---")
    st.markdown(
        "⚙️ Desenvolvido por **[Rafael Andrade](https://www.linkedin.com/in/rafaelsantoshome/)** com Streamlit, Pandas, Scikit-learn, Matplotlib e Seaborn.")
