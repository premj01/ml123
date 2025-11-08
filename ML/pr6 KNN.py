import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

k_values = range(1, 10)
accuracy = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy.append(accuracy_score(y_test, y_pred))

best_score = k_values[accuracy.index(max(accuracy))]
print(best_score)
print(max(accuracy))

plt.plot(k_values, accuracy, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("k-NN Classification: Effect of K (Iris Dataset)")
plt.show()

data = fetch_california_housing()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

k_vals = range(1, 10)
accuracy = []

for k in k_vals:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy.append(mean_squared_error(y_test, y_pred))
best_score = k_vals[accuracy.index(max(accuracy))]
print(best_score)
print(max(accuracy))

plt.plot(k_vals, accuracy, marker="o", color="orange")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean Squared Error")
plt.title("k-NN Regression: Effect of K (California Housing)")
plt.show()
