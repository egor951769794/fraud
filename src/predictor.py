import pickle as pkl

class Prediction:
    def __init__(self, filename):
        self.model = pkl.load(open(filename, 'rb'))

    def predict(self, X):
        X_clean = X[['time_diff_seconds_std', 'amount_std', 'address_Kfold_Target_Enc', 'is_passport_expired', 'same_passport', 'same_phone', 'operation_type_Kfold_Target_Enc', 'terminal_type_Kfold_Target_Enc']]
        return self.model.predict(X_clean)
    