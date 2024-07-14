from os import name
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Функция для загрузки данных с кэшированием
@st.cache(ttl=300)  # 300 секунд = 5 минут
def load_data():
    return pd.read_csv('archive_3.csv') 

# Функция для создания дашборда
def create_dashboard():
    st.title("Transaction Data Dashboard")
    
    # Общая информация о данных
    st.header("Представление датасета")
    st.write(df.head())
    
    # Основная статистика
    st.header("Основная статистика")
    st.write(df.describe())
    
    
    st.header("Распределение размеров транзакции")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['amount'], kde=True)
    st.pyplot(plt)


      # Список мошеннических транзакций
    st.header("Список мошеннических транзакций")
    fraudulent_transactions = df[df['is_fraud_pred'] == 1]
    st.write(fraudulent_transactions)

#круговая диаграмма
    st.header("Соотношение мошеннических транзакций к немошенническим")
    fraud_count = df['is_fraud_pred'].value_counts()
    labels = ['Non-Fraudulent', 'Fraudulent']
    sizes = [fraud_count[0], fraud_count[1]]
    colors = sns.color_palette('pastel')[0:len(labels)]
    explode = (0, 0.1)
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=140)
    plt.axis('equal') 
    st.pyplot(plt)

    #круговая диаграмма мошеннических транзакций
    st.header("Мошеннические транзакции по типу устройства")
    transaction_type_counts = fraudulent_transactions['terminal_type'].value_counts()
    labels = transaction_type_counts.index
    sizes = transaction_type_counts.values
    colors = sns.color_palette('pastel')[0:len(labels)]
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal') 
    st.pyplot(plt)

# Диаграмма с количеством транзакций с просроченными паспортами по городам
    st.header("Транзакции с просроченными паспортами по городам")
    expired_passport_transactions = df[df['is_passport_expired'] == True]
    city_counts = expired_passport_transactions['city'].value_counts()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=city_counts.index, y=city_counts.values, palette='viridis')
    plt.xlabel('Город')
    plt.ylabel('Количество транзакций')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

 # Точечная диаграмма зависимости промежутка между транзакциями от даты
    st.header("Зависимость промежутка между транзакциями от даты")
    plt.figure(figsize=(12, 8))

        # Данные для точек с не просроченными паспортами
    valid_passport_data = df[df['is_passport_expired'] == False]
    sns.scatterplot(
        data=valid_passport_data, 
        x='date_int', 
        y='time_diff_seconds_int', 
        hue='is_passport_expired', 
        size='amount', 
        sizes=(10, 100),  
        alpha=0.3,
        palette='coolwarm',
        legend=False,
        edgecolor=None
    )
    
    # Данные для точек с просроченными паспортами
    expired_passport_data = df[df['is_passport_expired'] == True]
    sns.scatterplot(
        data=expired_passport_data, 
        x='date_int', 
        y='time_diff_seconds_int', 
        hue='is_passport_expired', 
        size='amount', 
        sizes=(10, 100),  
        alpha=1.0,
        palette='bright',
        legend=False,
        edgecolor=None
    )
        
    plt.ylim(0, 600)  # Установка верхней границы для оси Y
    plt.xlabel('Дата')
    plt.ylabel('Временной промежуток между транзакциями в секундах')
    st.pyplot(plt)


 # Диаграмма с количеством транзакций с просроченными паспортами по городам
    st.header("Транзакции с просроченными паспортами по городам")
    expired_passport_transactions = df[df['is_passport_expired'] == True]
    plt.figure(figsize=(12, 8))
    sns.histplot(
        data=expired_passport_transactions,
        x="date_hour_int",
        hue="city",
        multiple="stack",
        palette="tab10",
        edgecolor=".3",
        linewidth=.5
    )
    plt.xlabel('Часы')
    plt.ylabel('Количество')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)


# Запуск дашборда

file_path = 'archive_3.csv'  # Замените на путь к вашему файлу
df = load_data()
create_dashboard()

# Проверка на изменения каждые 5 минут
if 'last_mod_time' not in st.session_state:
    st.session_state['last_mod_time'] = os.path.getmtime(file_path)

current_mod_time = os.path.getmtime(file_path)
if current_mod_time != st.session_state['last_mod_time']:
    st.session_state['last_mod_time'] = current_mod_time
    st.experimental_rerun()

st.experimental_memo(ttl=300)  # Установка кеширования на 5 минут
