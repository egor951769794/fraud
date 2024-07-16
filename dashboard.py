import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px

# Начальный DataFrame (можно оставить пустым)
df = pd.DataFrame()
st.set_page_config(
    layout="wide",
)
# Функция для обновления DataFrame
def update_df():
    global df
    # Замените на ваш код, который читает данные с устройства
    new_df = pd.read_csv('raw_data/archive_3.csv')  
    df = new_df

# Создание простого дашборда
st.title('Дашборд')

update_df()

col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.metric('All', len(df))
    st.metric('Volume', str(round(df['amount'].sum() / 1e9, 1)) + "М ₽")
    st.metric('Fraud', len(df.loc[df['is_fraud_pred'] == True]))

with col[1]:
    fig = px.scatter(df.loc[df['is_fraud_pred'] == True], x="date_int", y="time_diff_seconds_int", color="is_passport_expired")
    st.plotly_chart(fig, key="date_timediff")

    fig = px.histogram(df, x="city")
    st.plotly_chart(fig, key='city_count')

with col[2]:
    fig = px.pie(df, names='terminal_type', values='amount')
    st.plotly_chart(fig, key='terminal_count')
    
with col[2]:
    fig = px.pie(df, names='operation_type', values='amount')
    st.plotly_chart(fig, key='operation_count')

if st.button('Обновить'):
    update_df()
    st.table(df)