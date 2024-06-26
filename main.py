import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Dane wejściowe
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Dane wyjściowe
y = np.array([0, 1, 1, 0])

# Trenowanie modelu
clf = MLPClassifier(hidden_layer_sizes=(3),
                    activation='logistic',
                    learning_rate_init=1,
                    learning_rate='constant',
                    max_iter=1800,
                    n_iter_no_change=1000,
                    tol=1e-6,
                    batch_size='auto',
                    solver='sgd'
)


clf.fit(X, y.reshape(len(y)))

#accuracy=clf.score(X, y)

# Wykresy strat
plt.figure(figsize=(10, 5))
plt.plot(clf.loss_curve_)
plt.title('Losses in Each Layer')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

print('score:', clf.score(X, y))
print('predictions:', clf.predict(X))


errors = []

# Trenowanie modelu i zbieranie błędów klasyfikacji w każdej epoce
for i in range(clf.max_iter):
    clf.partial_fit(X, y, classes=np.unique(y))
    y_pred = clf.predict(X)
    mse = mean_squared_error(y, y_pred)
    error = 1 - accuracy_score(y, y_pred)
    if error >= 0.5:
        error = 1
    elif error < 0.5:
        error = 0

    errors.append(error)

# Wykresy błędu MSE
plt.figure(figsize=(10, 5))
plt.plot([mse] * len(y_pred))
plt.title('Mean Squared Error (MSE) in Each Layer')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()


# Wykres błędu klasyfikacji w każdej epoce uczenia
plt.figure(figsize=(10, 5))
plt.plot(errors)
plt.title('Classification Error in Each Epoch')
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
plt.show()

# Wykresy wag w obu warstwach
plt.figure(figsize=(10, 5))
plt.imshow(clf.coefs_[0], interpolation='none', cmap='viridis')
plt.colorbar()
plt.title('Weights of Hidden Layer')
plt.xlabel('Input Neurons')
plt.ylabel('Hidden Neurons')

plt.figure(figsize=(10, 5))
plt.imshow(clf.coefs_[1], interpolation='none', cmap='viridis')
plt.colorbar()
plt.title('Weights of Output Layer')
plt.xlabel('Hidden Neurons')
plt.ylabel('Output Neurons')
plt.show()