import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_decision_regions

# Wgranie danych
Xor = pd.read_csv('Xor_Dataset.csv')

# Wyjścia
y = Xor.iloc[0:10000, 2].values

# Wejścia
X = Xor[['X','Y']].to_numpy()

# Podział na zbiór trenowany i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Tworzenie modelu
clf = MLPClassifier(hidden_layer_sizes=(3),
                    activation='logistic',
                    learning_rate_init=1,
                    learning_rate='constant',
                    max_iter=5000,
                    n_iter_no_change=1000,
                    tol=1e-6,
                    solver='sgd'
                    )
# Trenowanie modelu
clf.fit(X_train, y_train)

# Wykresy strat
plt.figure(figsize=(10, 5))
plt.plot(clf.loss_curve_)
plt.title('Losses in Each Layer')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Making predictions on the training and test datasets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculating MSE for both datasets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Plotting MSE for both datasets
plt.figure(figsize=(10, 5))
plt.scatter([0, 1], [mse_train, mse_test], marker='o', s=100)
plt.title('Mean Squared Error (MSE) for Training and Test Data')
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.xticks([0, 1], ['Training', 'Test'])
plt.show()

batch_size = 20
classification_errors = []
num_epochs = len(X_train) // batch_size

for epoch in range(num_epochs):
    start = epoch * batch_size
    end = start + batch_size
    X_train_batch = X_train[start:end]
    y_train_batch = y_train[start:end]

    clf.fit(X_train_batch, y_train_batch)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the classification error
    error = np.mean(y_pred != y_test)
    if error >= 0.5:
        error = 1
    if error < 0.5:
        error = 0
    classification_errors.append(error)

plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), classification_errors)
plt.xlabel('Epoch')
plt.ylabel('Classification Error')
plt.title('Classification Error of MLPClassifier')
plt.show()

"""
plot_decision_regions(X_train, y_train, clf=clf)
plt.title('Wages')
plt.xlabel(' ')
plt.ylabel('Output Layer')
plt.show()

plot_decision_regions(X_test, y_test, clf=clf)
plt.title('Wages')
plt.xlabel(' ')
plt.ylabel('Output Layer')
plt.show()
"""