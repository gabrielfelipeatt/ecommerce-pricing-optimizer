# 📊 E-Commerce Pricing Optimizer: Inteligência Preditiva & Performance

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3-green.svg)](https://pycaret.org/)

## 🎯 Contexto do Projeto
No cenário de e-commerce, a definição de preços impacta diretamente a margem de lucro e o volume de vendas. Este projeto utiliza o dataset da **Olist (Brazilian E-Commerce)** para construir uma ferramenta de suporte à decisão, permitindo que gestores comerciais visualizem a saúde do catálogo e simulem preços otimizados através de Machine Learning.

O diferencial desta solução é a união da análise estatística clássica (**Curva ABC**) com a modelagem preditiva moderna (**Regressão via AutoML**).

## 🚀 Funcionalidades Principais

### 1. Análise Estratégica de Portfólio (Curva ABC)
Classificação automatizada de categorias de produtos baseada na receita gerada (Teorema de Pareto).
- **Classe A:** Itens críticos que geram 80% do faturamento.
- **Classe B/C:** Itens de cauda longa que exigem estratégias de giro de estoque diferenciadas.

### 2. Modelagem Preditiva com AutoML
Utilização da biblioteca **PyCaret** para treinar e comparar diversos algoritmos de regressão (como Random Forest, XGBoost e LightGBM). 
- O modelo seleciona automaticamente o algoritmo com o menor **MAE (Mean Absolute Error)** e maior **R²**, garantindo previsões confiáveis.

### 3. Simulador de Pricing em Tempo Real
Interface interativa onde o usuário pode ajustar variáveis como **peso do produto**, **mês da venda** e **sazonalidade** para obter instantaneamente uma sugestão de preço baseada no comportamento histórico do mercado.



## 🛠️ Stack Técnica
- **Linguagem:** Python 3.11
- **Dashboard:** Streamlit (UX otimizada com Cache e Session State)
- **Data Prep:** Pandas (Tratamento de outliers via Quantis)
- **Machine Learning:** PyCaret & Scikit-Learn
- **Visualização:** Plotly (Gráficos interativos)

## 📂 Como Executar este Projeto

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/gabrielfelipeatt/ecommerce-pricing-optimizer.git](https://github.com/gabrielfelipeatt/ecommerce-pricing-optimizer.git)