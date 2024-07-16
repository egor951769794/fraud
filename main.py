from pymongo import MongoClient

import threading

from bunnet import Document, Indexed, init_bunnet
import os
import time
from sys import argv
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from pymongo.errors import DuplicateKeyError
from datetime import datetime, timedelta
from src.predictor import Prediction
import torch


import pandas as pd

from src.preprocessor import Preprocessing
from src.estimator import Estimator
from src.fraud_finder_algorithm import FraudFinder


class Record(Document):
    id_transaction: Indexed(int, unique=True)
    date: datetime
    card: object
    client: object
    date_of_birth: datetime
    passport: object
    passport_valid_to: object
    phone: object
    operation_type: object
    amount: float
    operation_result: object
    terminal_type: object
    city: object
    address: object
    date_int: int
    dob_int: int
    time_diff_seconds: timedelta
    time_diff_seconds_int: int
    date_hour_int: int
    date_day_int: int
    amount_std: float
    is_high_amount: bool
    time_diff_seconds_std: float
    is_passport_expired: bool
    prev_passport: object
    same_passport: bool
    prev_phone: object
    same_phone: bool
    fair_docs: bool
    address_Kfold_Target_Enc: float
    operation_type_Kfold_Target_Enc: float
    terminal_type_Kfold_Target_Enc: float
    labels: float
    labels_std: float
    is_fraud: bool



def get_df(path):
    success = False
    while not success:
        try:
            sep = ''
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        line1 = line.replace('id_transaction', '')
                        sep = line1[0]
                        if line1[1] != 'd':
                            raise Exception("\nОшибка: неверный формат файла")
                    if i > 0:
                        break

            df = pd.read_csv(path, sep=sep)
            print(df.head(5))
            if not df.empty:
                success = True
        except IOError:
            print("\nОжидание загрузки файла...")
            time.sleep(1)

    if set([
        'id_transaction',
        'date',
        'card',
        'client',
        'date_of_birth',
        'passport',
        'passport_valid_to',
        'phone',
        'operation_type',
        'amount',
        'operation_result',
        'terminal_type',
        'city',
        'address'
    ]).issubset(df.columns):
        return df
    else:
        raise Exception("\nОшибка: Данные некорректны")
    
    
def data_save_subthread(df, y):
    print("\nПодключение к БД...")
    client = MongoClient("mongodb://localhost:27017")
    init_bunnet(database=client.transactions, document_models=[Record])
    print("\nПодключение к БД: успешно")

    df['is_fraud'] = y
    print("\nСохранение данных в БД...")
    
    for i, record in df.iterrows():
        new_record = Record(
            id_transaction = str(record['id_transaction']),
            date = record['date'],
            card = record['card'],
            client = record['client'],
            date_of_birth = record['date_of_birth'],
            passport = record['passport'],
            passport_valid_to = record['passport_valid_to'],
            phone = record['phone'],
            operation_type = record['operation_type'],
            amount = record['amount'],
            operation_result = record['operation_result'],
            terminal_type = record['terminal_type'],
            city = record['city'],
            address = record['address'],
            date_int = record['date_int'],
            dob_int = record['dob_int'],
            time_diff_seconds = record['time_diff_seconds'],
            time_diff_seconds_int = record['time_diff_seconds_int'],
            date_hour_int = record['date_hour_int'],
            date_day_int = record['date_day_int'],
            amount_std = record['amount_std'],
            is_high_amount = record['is_high_amount'],
            time_diff_seconds_std = record['time_diff_seconds_std'],
            is_passport_expired = record['is_passport_expired'],
            prev_passport = record['prev_passport'],
            same_passport = record['same_passport'],
            prev_phone = record['prev_phone'],
            same_phone = record['same_phone'],
            fair_docs = record['fair_docs'],
            address_Kfold_Target_Enc = record['address_Kfold_Target_Enc'],
            operation_type_Kfold_Target_Enc = record['operation_type_Kfold_Target_Enc'],
            terminal_type_Kfold_Target_Enc = record['terminal_type_Kfold_Target_Enc'],
            labels = record['labels'],
            labels_std = record['labels_std'],
            is_fraud = record['is_fraud'],
        )

        try:
            new_record.insert()
        except DuplicateKeyError:
            print("\nОшибка первичного ключа")
    
    print("Данные сохранены")

    
def model_train_subthread(filename, X, y):
    print("\nОбучение модели...")
    estimator = Estimator(filename)
    estimator.train(X, y)
    print("\nОбучение завершено")

def update_dashboard_subthread(df, dashboard):
    dashboard.update_df(df)
    

def process_file(filename, df):
    with torch.no_grad():
        X = df
        X_pred = X[['time_diff_seconds_std', 'amount_std', 'address_Kfold_Target_Enc', 'is_passport_expired', 'same_passport', 'same_phone', 'operation_type_Kfold_Target_Enc', 'terminal_type_Kfold_Target_Enc']]
        y_pred = Prediction(filename).predict(X_pred).round().detach().numpy()
        y = FraudFinder().calculate_frauds(X)
        db_save_thread = threading.Thread(target=data_save_subthread, args=(X, y_pred,))
        db_save_thread.start()
        if do_train:
            train_thread = threading.Thread(target=model_train_subthread, args=(filename, X, y,))
            train_thread.start()
        
        X['is_fraud_pred'] = y_pred
        # run_dashboard_thread = threading.Thread(target=update_dashboard_subthread, args=(X, dashboard))
        # run_dashboard_thread.start()
    
def file_handling_subthread(path):
    print(f"\nПолучен файл {path}")
    if path.endswith('.csv'):
        print("\nОбработка данных...")
        try:
            df = get_df(path)
            prepros = Preprocessing(verbosity=True)
            print("\nПреобразование...")
            df_processed = prepros.transform(df)
            process_file(os.path.dirname(__file__) + "\\src\\compiled\\nn_model.pth.tar", df_processed)
            
        except Exception as e:
            print(e)

    else:
        print("\nОшибка: неверный формат файла")


def on_create(event):
    path = os.path.dirname(__file__) + "\\raw_data\\" + event.src_path.split("\\")[1]

    thread = threading.Thread(target=file_handling_subthread, args=(path,))
    thread.start()

if __name__ == "__main__":
    do_train = len(argv) > 1 and argv[1] == 'do-train'
    path = './raw_data'
    event_handler = PatternMatchingEventHandler(["*"], None, False, True)
    event_handler.on_created = on_create
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()