# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def run():
        
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
    
    
    #Importação base de dados
    df_orig = pd.read_csv('winequality-red.csv')
    
    #Remoção de linhas duplicadas
    df1= df_orig
    df1.drop_duplicates(inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df = df1.copy()
    
    
    #Exibir Dataset no Streamlit
    
    st.title('Projeto de Regressão Linear para Predição da Qualidade de Vinhos')
    st.subheader('Descrição')
    st.write('Existem inúmeros fatores que podem afetar a qualidade e o valor comercial do vinho. Até mesmo fatores climáticos do ano da safra podem causar alterações químicas que afetarão sua qualidade, tornando a precificação adequada um grande desafio. Neste caso específico o Dataset utilizado para treino provê dados sobre vinhos tintos.  A partir de fatores químicos iremos tentar prever a qualidade do vinho (fator sensorial)')
    st.subheader('Dataset (Amostra)')
    st.write(df.head(50))
    
    #Tratamento dos dados
    st.subheader('Pré Tratamento dos Dados')
    df['quality_orig'] = df['quality']
    df['quality'].replace({3:4,8:7},inplace=True)
    
    
    #Gráficos 
    st.info('Devido à baixa representatividade dos vinhos com nota 3 e 4 e os com nota 7 e 8, optei por os agrupar, ficando com 4 possíveis classificações (3+4, 5, 6, 7+8)')
    fig, ax = plt.subplots(1,2, figsize=(8, 4))
    sns.countplot(data=df, x='quality_orig',  ax=ax[0])
    ax[0].set_title('Distribuição Original do Target')
    
    sns.countplot(data=df, x='quality',  ax=ax[1])
    ax[1].set_title('Distribuição Agrupada')
    plt.tight_layout()
    st.pyplot(fig)
    
    #Histogramas e Boxplots
    st.subheader('Análise de Distribuição e Outliers das Variáveis Independentes')
    st.info('Ao plotar o histograma podemos notar que as variáveis "citric acid", "free sulfur dioxide", "total sulfur dioxide" e "alcohool", não possuem distribuição normal.')
    st.info('Após analisar os outliers através dos gráficos abaixo, os adequei ao limite inferior ou superior de cada faixa do target. Se um outlier fosse maior que o limite superior, substituí pelo limite superior, se fosse menor o limite inferior, substituí pelo limite inferior.')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fig, ax = plt.subplots(11,2, figsize=(10, 40))
    for i, col in enumerate(df.columns[0:11]):
        ax[i,0].hist(df[col], bins=15, edgecolor='black')
        ax[i, 0].set_title(f'Histograma de {col}')
        ax[i, 0].set_xlabel(col)
        ax[i, 0].set_ylabel('Frequência')
    
        sns.boxplot(data = df, y=col, x = 'quality',ax=ax[i,1])
        ax[i, 1].set_title(f'Boxplot de {col}')
        ax[i, 1].set_xlabel(col)
        ax[i, 1].set_ylabel('Valor')
    plt.tight_layout()
    st.pyplot(fig)
    
    
    #Retirando Outliers
    df_norm = pd.DataFrame()
    
    for fatia in (4,5,6,7):
        df_slice = df.loc[df['quality'] == fatia].copy()
        df_slice.reset_index(drop=True, inplace=True)
        for col in df_slice.columns:
            Q1 = df_slice[col].quantile(0.25)
            Q3 = df_slice[col].quantile(0.75)
            IQR = Q3-Q1
    
            lim_inf = Q1 - (1.5 * IQR)
            lim_sup = Q3 + (1.5 * IQR)
    
            for i in range(len(df_slice[col])):
                if df_slice.loc[i, col] < lim_inf:
                    df_slice.loc[i, col] = lim_inf
    
                if df_slice.loc[i, col] > lim_sup:
                    df_slice.loc[i, col] = lim_sup
        df_norm = pd.concat([df_norm,df_slice], axis=0)
    
    df_norm.reset_index(drop=True, inplace=True)
    
    #Matriz correlação
    st.info('Através da Matriz de correlação, pude observar que não há variáveis altamente correlacionadas (>=0.8). Observei também que o açúcar residual não possui correlação com a qualidade. As variáveis com maior interferência na qualidade são o álcool, sulfatos e acidez volátil.')
    
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    cor_matrix = df_norm.corr()
    half_matrix = np.triu(cor_matrix)
    sns.heatmap(cor_matrix, annot=True, fmt='.1f', vmin=-1, vmax=1, cmap='RdBu', mask=half_matrix, ax=ax)
    st.pyplot(fig)
    
    df_norm.drop(['fixed acidity','residual sugar','free sulfur dioxide','pH',], axis=1, inplace=True)
    
    x = df_norm.iloc[:,:6].values #Variáveis independentes
    y = df_norm.iloc[:,6:7].values #Variável Target
    
    #70% para treino, 30% para teste
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    MinMax_scaler = MinMaxScaler()
    
    x_train = MinMax_scaler.fit_transform(x_train)
    x_test = MinMax_scaler.transform(x_test)
    
    model = LinearRegression()
    model.fit(x_train,y_train)
    
    y_hat = model.predict(x_test)
    
    # Cálculos
    mape = mean_absolute_percentage_error(y_test, y_hat)
    mae = mean_absolute_error(y_test, y_hat)
    rmse = np.sqrt(mean_squared_error(y_test, y_hat))
    mse = mean_squared_error(y_test, y_hat)
    
    # Exibição dos resultados
    st.subheader('Métricas de Desempenho do Modelo')
    st.warning(f'MAPE: {mape:.2f}%')
    st.warning(f'MAE: {mae:.2f}')
    st.warning(f'RMSE: {rmse:.2f}')
    st.warning(f'MSE: {mse:.2f}')
    

if __name__ == "__main__":
    run()
