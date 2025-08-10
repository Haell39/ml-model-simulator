import streamlit as st
import pandas as pd

#  Configuração da Página
st.set_page_config(
    page_title="Simulador de Modelos de ML",
    page_icon="🤖",
    layout="wide"
)

#  Título e Descrição
st.title("🤖 Simulador Interativo de Modelos de Machine Learning")
st.write(
    "Bem-vindo ao seu simulador de Machine Learning! "
    "Esta ferramenta permite que você carregue seus próprios dados, "
    "escolha um modelo, treine-o e avalie sua performance de forma rápida e intuitiva."
)
st.write("---")

#  Barra Lateral (Sidebar)
with st.sidebar:
    st.header("1. Carregue seus Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")


    # Dicionário para armazenar os parâmetros do modelo
    params = {}

    # Só mostra as opções de modelo se um arquivo for carregado
    if uploaded_file is not None:
        st.header("2. Escolha o Modelo e Ajuste os Parâmetros")

        # Lista de modelos disponíveis
        model_type = st.selectbox(
            "Selecione o Modelo",
            ("Regressão Logística", "Árvore de Decisão", "Random Forest", "SVM")
        )

        # Lógica para mostrar parâmetros específicos de cada modelo
        if model_type == "Regressão Logística":
            st.subheader("Hiperparâmetros do Modelo")
            params['C'] = st.slider("Parâmetro de Regularização (C)", 0.01, 10.0, 1.0)
            params['max_iter'] = st.slider("Máximo de Iterações", 100, 1000, 100)

        elif model_type == "Árvore de Decisão":
            st.subheader("Hiperparâmetros do Modelo")
            params['max_depth'] = st.slider("Profundidade Máxima da Árvore", 2, 30, 10, key='max_depth_dt')
            params['criterion'] = st.selectbox("Critério de Divisão", ("gini", "entropy"), key='criterion_dt')

        elif model_type == "Random Forest":
            st.subheader("Hiperparâmetros do Modelo")
            params['n_estimators'] = st.slider("Número de Árvores (n_estimators)", 10, 500, 100, key='n_estimators_rf')
            params['max_depth'] = st.slider("Profundidade Máxima da Árvore", 2, 30, 10, key='max_depth_rf')
            params['criterion'] = st.selectbox("Critério de Divisão", ("gini", "entropy"), key='criterion_rf')

        elif model_type == "SVM":
            st.subheader("Hiperparâmetros do Modelo")

            params['C'] = st.slider("Parâmetro de Regularização (C)", 0.01, 10.0, 1.0, key='C_svm')

            params['kernel'] = st.selectbox("Kernel", ("linear", "rbf", "poly"), key='kernel_svm')



#  Lógica Principal
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    st.success("Arquivo carregado com sucesso!")

    st.subheader("Pré-visualização dos Dados")
    st.dataframe(df.head())

    st.subheader("Configuração das Variáveis do Modelo")

    colunas = df.columns.tolist()

    target_variable = st.selectbox(
        "Selecione a sua variável-alvo (a coluna que você quer prever):",
        options=colunas,
        key='target'
    )

    features_disponiveis = [col for col in colunas if col != target_variable]

    feature_variables = st.multiselect(
        "Selecione as suas variáveis preditoras (features):",
        options=features_disponiveis,
        default=features_disponiveis,
        key='features'
    )

    st.write("---")
    st.write("Você selecionou:")
    st.info(f"**Variável-Alvo (y):** `{target_variable}`")
    st.info(f"**Variáveis Preditoras (X):** `{feature_variables}`")

    # Exibe os parâmetros selecionados para o usuário (feedback)
    if params:
        st.write("Configurações do Modelo Escolhido:")
        st.json(params)  # st.json --> dicionário de forma bonita

else:
    st.warning("Por favor, carregue um arquivo CSV para começar.")