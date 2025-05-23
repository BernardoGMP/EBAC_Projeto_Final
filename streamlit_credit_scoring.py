
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Credit Scoring App", layout="wide")

st.title("Credit Scoring - Projeto Final")

# Função de pré-processamento
def preprocessamento(df):
    # Remover colunas desnecessárias, se existirem
    cols_remover = ['index', 'data_ref']
    df = df.drop(columns=[col for col in cols_remover if col in df.columns], errors='ignore')

    # Identificar tipos de variáveis
    variaveis_qualitativas = [col for col in df.columns if df[col].nunique() <= 6]
    variaveis_quantitativas = [col for col in df.columns if col not in variaveis_qualitativas]

    # Preencher nulos
    for col in variaveis_quantitativas:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in variaveis_qualitativas:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Substituição de outliers (IQR) pela média
    for col in variaveis_quantitativas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        media = df[col].mean()
        df[col] = df[col].apply(lambda x: media if x < limite_inferior or x > limite_superior else x)

    # Dummies
    df = pd.get_dummies(df, columns=variaveis_qualitativas, drop_first=True)

    return df

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Faça o upload do arquivo CSV para escoragem", type=["csv"])

# Carregar modelo treinado
@st.cache_resource
def load_model():
    with open("model_final.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    st.subheader("Pré-visualização dos dados carregados")
    st.dataframe(df_input.head())

    # Aplicar pré-processamento
    try:
        df_processed = preprocessamento(df_input)

        # Alinhar colunas com as do modelo, preenchendo faltantes com 0
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[model_features]

        # Aplicar o modelo
        predictions = model.predict_proba(df_processed)[:, 1]
        df_output = df_input.copy()
        df_output["score"] = predictions

        st.subheader("Resultado da Escoragem")
        st.dataframe(df_output.head())

        # Download dos resultados
        csv = df_output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar resultado como CSV",
            data=csv,
            file_name="resultado_escoragem.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Ocorreu um erro ao escorar os dados: {e}")
else:
    st.info("Por favor, envie um arquivo CSV para iniciar a escoragem.")
