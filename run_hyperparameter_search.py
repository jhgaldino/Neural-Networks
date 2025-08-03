
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
import random

# Load and preprocess data
df = pd.read_csv('E0.csv')
df = df.drop(['Div', 'Date', 'HTR', 'HTHG', 'HTAG', 'Referee'], axis=1)
df = df.iloc[:, :-39]
dfv2 = df.replace({"H":1,"D":0,"A":-1})
dfv3 = dfv2.iloc[:, 2:]
NN = dfv3[['FTR']]
input_data = dfv3.iloc[:, 1:]
output_data = NN

for e in range(len(input_data.columns)):
    max_val = input_data.iloc[:, e].max()
    if max_val < 10:
        input_data.iloc[:, e] /= 10
    elif max_val < 100:
        input_data.iloc[:, e] /= 100
    else:
        print("Error in normalization! Please check!")

X = input_data.values
y = output_data.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_dist = {
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'batch_size': [10, 20, 30],
    'epochs': [50, 100, 150],
    'neurons': [8, 12, 16],
    'hidden_layers': [1, 2, 3],
    'lr': [0.001, 0.01, 0.1],
    'momentum': [0.0, 0.5, 0.9]
}

# Manual Randomized Search
best_score = -1
best_params = {}

print("Starting hyperparameter search...")

for i in range(10): # Number of iterations
    params = {k: random.choice(v) for k, v in param_dist.items()}
    print(f"Testing params: {params}")

    model = Sequential()
    model.add(Dense(params['neurons'], input_dim=14, activation='relu'))
    for _ in range(params['hidden_layers'] - 1):
        model.add(Dense(params['neurons'], activation='relu'))
    model.add(Dense(1, activation='tanh'))

    optimizer_name = params['optimizer']
    lr = params['lr']
    momentum = params['momentum']

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr, momentum=momentum)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=momentum)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

    score = model.evaluate(X_test, y_test, verbose=0)[1] # Accuracy

    if score > best_score:
        best_score = score
        best_params = params

print(f"Best score: {best_score}")
print(f"Best parameters found: {best_params}")
