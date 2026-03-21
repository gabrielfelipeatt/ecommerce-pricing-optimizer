import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, predict_model
import os

# Configuração da Página
st.set_page_config(page_title="Data Science Project - Pricing Hub", layout="wide")

# --- ESTADO DA SESSÃO ---
# Isso garante que os dados não sumam ao mexer no simulador
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def load_and_clean_data(path):
    # Otimização: ler apenas colunas necessárias para economizar memória
    items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
    orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    
    df = items.merge(products, on='product_id').merge(orders, on='order_id')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['month'] = df['order_purchase_timestamp'].dt.month
    df['day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    
    # Tratamento Sênior: Remover preços absurdos (Outliers) para melhorar o modelo
    q_low = df["price"].quantile(0.01)
    q_hi  = df["price"].quantile(0.95)
    df = df[(df["price"] < q_hi) & (df["price"] > q_low)]
    
    return df.dropna(subset=['product_category_name'])

@st.cache_resource
def train_model(data):
    # Session_id fixo garante que o experimento seja reproduzível
    s = setup(data, target='price', verbose=False, html=False, session_id=123, 
              remove_outliers=True, transformation=True)
    best_model = compare_models(n_select=1)
    results = pull()
    return best_model, results

# --- INTERFACE ---
st.title("📊 Pricing Hub: Inteligência de Dados E-commerce")
st.markdown(f"**Cientista de Dados:** Gabriel Felipe Andrade Lima")

# Sidebar
default_path = r"C:\Users\Gabriel\.cache\kagglehub\datasets\olistbr\brazilian-ecommerce\versions\2"
path_input = st.sidebar.text_input("Diretório dos Dados", default_path)

if st.sidebar.button("Carregar e Processar"):
    st.session_state.df = load_and_clean_data(path_input)
    st.session_state.data_loaded = True

# Só executa o restante se os dados estiverem carregados no session_state
if st.session_state.data_loaded:
    df = st.session_state.df
    
    # 1. CURVA ABC
    st.header("1. Curva ABC por Categoria")
    abc = df.groupby('product_category_name')['price'].sum().reset_index().sort_values(by='price', ascending=False)
    abc['rev_cum'] = abc['price'].cumsum()
    abc['percentage'] = (abc['rev_cum'] / abc['price'].sum()) * 100
    abc['Class'] = abc['percentage'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
    
    fig_abc = px.bar(abc, x='product_category_name', y='price', color='Class', template="plotly_white")
    st.plotly_chart(fig_abc, use_container_width=True)

    # 2. MODELAGEM
    st.header("2. 🤖 Inteligência Artificial - Regressão")
    data_modelling = df[['price', 'month', 'day_of_week', 'product_weight_g', 'product_description_lenght']].sample(2000, random_state=42)
    
    model, metrics = train_model(data_modelling)
    st.dataframe(metrics)

    # 3. SIMULADOR
    st.header("3. Simulador de Pricing")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        w = st.number_input("Peso (g)", value=500)
        m = st.slider("Mês", 1, 12, 6)
        
    input_df = pd.DataFrame([[m, 0, w, 500]], columns=['month', 'day_of_week', 'product_weight_g', 'product_description_lenght'])
    pred = predict_model(model, data=input_df)
    
    with c2:
        st.metric("Preço Sugerido", f"R$ {pred['prediction_label'][0]:.2f}")
        st.write("Lógica: O modelo analisa a sazonalidade e o custo logístico atrelado ao peso.")