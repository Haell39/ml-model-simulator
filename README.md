# 🚀 Insight Navigator: Análise e Modelagem Preditiva

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://insight-navigator.streamlit.app/)

Uma aplicação web completa para o ciclo de vida de projetos de Ciência de Dados, construída com Streamlit. O **Insight Navigator** permite que usuários carreguem seus próprios datasets, realizem uma Análise Exploratória de Dados (EDA) detalhada e, em seguida, treinem, comparem e avaliem múltiplos modelos de Machine Learning de forma interativa.

---

## ✨ Principais Funcionalidades

A ferramenta é dividida em dois módulos principais:

### 📊 Análise Exploratória (EDA)
- **Visão Geral:** Exibição das primeiras linhas e estatísticas descritivas do dataset.
- **Análise de Distribuição:** Geração de histogramas interativos para qualquer variável numérica.
- **Matriz de Correlação:** Mapa de calor para visualizar a relação entre as variáveis.
- **Análise Categórica:** Gráficos de contagem para entender a frequência de cada classe.
- **Gráficos Bivariados:** Scatter plots e Box plots para explorar a relação entre features.

### 🤖 Simulador de Modelos
- **Seleção de Modelos:** Escolha entre Regressão Logística, Árvore de Decisão e Random Forest.
- **Ajuste de Hiperparâmetros:** Controles interativos na barra lateral para ajustar os parâmetros do modelo em tempo real.
- **Avaliação Completa:** Geração de resultados detalhados em abas, incluindo:
  - **Métricas:** Acurácia, Precisão, Recall e F1-Score.
  - **Matriz de Confusão:** Visualização clara dos acertos e erros do modelo.
  - **Curva ROC:** Análise da performance do classificador com cálculo de AUC.
  - **Comparativo:** Tabela de performance comparando os modelos disponíveis.
  - **Análise de Probabilidades:** Tabela com as probabilidades de previsão para cada amostra de teste.

---

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Streamlit:** Para a construção da interface web interativa.
- **Pandas:** Para manipulação e limpeza dos dados.
- **Scikit-learn:** Para pré-processamento, modelagem e avaliação.
- **Seaborn & Matplotlib:** Para a criação dos gráficos e visualizações.

---

## ⚙️ Como Executar o Projeto Localmente

Siga os passos abaixo para rodar a aplicação no seu computador.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    # Para Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # Para Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```

A aplicação abrirá automaticamente no seu navegador.

---

## 👨‍💻 Autor

Desenvolvido por **[Rafael Andrade](https://www.linkedin.com/in/rafaelsantoshome/)**.