import streamlit as st
import pandas as pd
import time
import plotly.express as px

class Dashboard:

    def __init__(self):
        self.df = pd.DataFrame()
    
    # Функция для обновления DataFrame
    def update_df(self, df):
        self.df.append(df, ignore_index=True)

    def create_dashboard(self):
        col = st.columns((1.5, 4.5, 2), gap='medium')

        with col[0]:
            st.metric('All', len(self.df))
            st.metric('Volume', str(round(self.df['amount'].sum() / 1e9, 1)) + "М ₽")
            st.metric('Fraud', len(self.df.loc[self.df['is_fraud_pred'] == True]))

        with col[1]:
            fig = px.scatter(self.df.loc[self.df['is_fraud_pred'] == True], x="date_int", y="time_diff_seconds_int", color="is_passport_expired", size="amount")
            st.plotly_chart(fig, key="date_timediff")

            fig = px.histogram(self.df.loc[self.df['is_fraud_pred'] == True], x="city")
            st.plotly_chart(fig, key='city_count')

        with col[2]:
            fig = px.pie(self.df.loc[self.df['is_fraud_pred'] == True], values='amount', names='city', title='Fraud among cities')

            

        if st.button('Обновить'):
            self.update_df()