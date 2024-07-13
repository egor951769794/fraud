from sklearn.base import BaseEstimator, TransformerMixin
from src.kfold import KFoldTargetEncoderTrain
from scipy import stats
import numpy as np
import pandas as pd

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, verbosity=False):
        self.verbosity = verbosity
        pd.options.mode.chained_assignment = None

    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])

        df['date_int'] = (df['date'].astype(np.int64) // (10 ** 9)).astype('Int32')
        df['dob_int'] = (df['date_of_birth'].astype(np.int64) // (10 ** 9)).astype('Int32')

        df['time_diff_seconds'] = df.groupby('client')['date'].diff()
        df['time_diff_seconds_int'] = df['time_diff_seconds'].dt.seconds.astype('Int32')

        df['date_hour_int'] = df['date'].dt.hour.astype('Int8')
        df['date_day_int'] = df['date'].dt.day.astype('Int8')

        df = df.dropna()

        df['amount_std'] = stats.zscore(df['amount'].astype(np.float64))

        df['is_high_amount'] = df['amount_std'] > 1

        df['time_diff_seconds_std'] = stats.zscore(df['time_diff_seconds_int'].astype(np.float64))

        df['is_passport_expired'] = df.apply(is_passport_expired, axis=1)

        df['prev_passport'] = df.groupby('client')['passport'].shift()
        df['same_passport'] = df.apply(lambda x: True if x.isna()['prev_passport'] else x['prev_passport'] == x['passport'], axis=1)

        df['prev_phone'] = df.groupby('client')['phone'].shift()
        df['same_phone'] = df.apply(lambda x: True if x.isna()['prev_phone'] else x['prev_phone'] == x['phone'], axis=1)

        df['fair_docs'] = df.apply(lambda x: x['same_phone'] and x['same_passport'], axis=1)

        print("KFold...")

        targetc = KFoldTargetEncoderTrain('address', 'fair_docs', n_fold=7)
        df = targetc.fit_transform(df)
        df['address_Kfold_Target_Enc'] = df['address_Kfold_Target_Enc'].apply(lambda x: x * 100)

        targetc = KFoldTargetEncoderTrain('operation_type', 'fair_docs', n_fold=100)
        df = targetc.fit_transform(df)

        targetc = KFoldTargetEncoderTrain('terminal_type', 'fair_docs', n_fold=60)
        df = targetc.fit_transform(df)

        # df.drop(['prev_phone', 'prev_passport'], inplace=True)
        # print("13231321")

        if self.verbosity:
            print(df.info())

        return df


def is_passport_expired(row):
    if row['passport_valid_to'] == 'бессрочно':
        return False
    return pd.to_datetime(row['passport_valid_to']) < row['date']
