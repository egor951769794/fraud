from sklearn.cluster import OPTICS
import pandas as pd
from scipy import stats

class FraudFinder:
    def __init__(self, eps=0.1, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples

    def calculate_frauds(self, X):
        db = OPTICS(min_samples=self.min_samples, eps=self.eps).fit(X[['time_diff_seconds_std', 'amount_std', 'address_Kfold_Target_Enc', 'is_passport_expired', 'same_passport', 'same_phone', 'operation_type_Kfold_Target_Enc', 'terminal_type_Kfold_Target_Enc']])
        labels = pd.Series(db.labels_)
        X['labels'] = labels
        X['labels_std'] = stats.zscore(X['labels'])

        fraud_count = []
        fraud_freq = 0
        fraud_index = 0
        count = []

        i = 1
        while X.loc[(X['labels_std'] > i)].shape[0] > 0:
            fraud_count.append(X.loc[(X['labels_std'] > i) & (X['fair_docs'] == False)].shape[0])
            count.append(X.loc[(X['labels_std'] > i)].shape[0])

            if fraud_count[-1] / count[-1] > fraud_freq:
                fraud_index = i
                fraud_freq = fraud_count[-1] / count[-1]

            i += 0.01

        is_fraud = X.apply(lambda x: x['labels_std'] > fraud_index, axis=1)
        return is_fraud
