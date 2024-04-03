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
clf = MLPClassifier(hidden_layer_sizes=(5),
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


# Predykcja dla zbioru trenowanego i testowanego
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Obliczanie błędu MSE dla obu zbiorów
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Wykres MSE
plt.figure(figsize=(10, 5))
plt.scatter([0, 1], [mse_train, mse_test], marker='o', s=100)
plt.title('Mean Squared Error for Training and Test Data')
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.xticks([0, 1], ['Training', 'Test'])

missclasifications = []
errors = []

# Trenowanie modelu i zbieranie błędów klasyfikacji w każdej epoce
for i in range(clf.n_iter_):
    clf.partial_fit(X_train, y_train, classes=np.unique(y))
    y_pred = clf.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    error = 1 - accuracy_score(y_train, y_pred)
    missclasification = error
    if error >= 0.5:
        error = 1
    if error < 0.5:
        error = 0
    missclasifications.append(missclasification)
    errors.append(error)

# Wykres błędu klasyfikacji w każdej epoce uczenia
plt.figure(figsize=(10, 5))
plt.plot(errors)
plt.title('Classification Error in Each Epoch')
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
plt.show()

# Wykres błędu klasyfikacji w każdej epoce uczenia
plt.figure(figsize=(10, 5))
plt.plot(missclasifications)
plt.title('Classification Error in Each Epoch')
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
plt.show()


output_layer_weights = clf.coefs_[0]
# Warstwa wyjściowa
plt.figure(figsize=(10, 5))
plt.imshow(output_layer_weights.T, cmap='viridis', interpolation='nearest')
plt.title('Weights Leading to the Output Layer coefs_')
plt.ylabel('Output Neurons')
plt.xlabel('Hidden Neurons')
plt.colorbar()
plt.show()

# Tworzenie siatki punktów
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Wagi przedstawione poprzez przestrzeń decyzyjną
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Space')
plt.show()

# Wagi na podstawie regionów granicznych
plot_decision_regions(X_train, y_train, clf=clf)
plt.title('Train decision regions')
plt.xlabel(' ')
plt.ylabel('Output Layer')
plt.show()

plot_decision_regions(X_test, y_test, clf=clf)
plt.title('Test decision regions')
plt.xlabel(' ')
plt.ylabel('Output Layer')
plt.show()
