from src.deep_model import Deep
import torch
from torch.optim import Adam
import numpy as np
import torch.nn as nn
import copy
from sklearn.model_selection import train_test_split

class Estimator:
    def __init__(self, filename):
        self.filename = filename
        self.model = Deep()
        state = torch.load(filename)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_acc = state['acc']
        self.epoch = state['epoch']
        self.loss_fn = nn.BCELoss()

    def train(self, X, y):
        X = X[[
            'time_diff_seconds_std', 
            'amount_std', 
            'address_Kfold_Target_Enc', 
            'is_passport_expired', 
            'same_passport', 
            'same_phone', 
            'operation_type_Kfold_Target_Enc', 
            'terminal_type_Kfold_Target_Enc']]
        
        X = torch.tensor(X.astype(np.float32).values)
        y = torch.tensor(y.astype(np.float32).values)


        # split train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        n_epochs = 25    # number of epochs to run
        batch_size = 60  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)

        best_weights = None
        for epoch in range(self.epoch, self.epoch + n_epochs):
            self.model.train()
            for start in batch_start:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                y_batch = y_batch.unsqueeze(1)
                # forward pass
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()
            
            self.model.eval()
            y_pred = self.model(X_test)
            acc = (y_pred.round() == y_test).float().mean()
            acc = float(acc)
            if acc > self.best_acc:
                self.best_acc = acc
                best_weights = copy.deepcopy(self.model.state_dict())
        # restore model and return best accuracy
        self.model.load_state_dict(best_weights)

        state = {'epoch': self.epoch + n_epochs, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'acc': self.best_acc}
        torch.save(state, self.filename)