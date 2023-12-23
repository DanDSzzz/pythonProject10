import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.utils import shuffle

data = pd.read_csv("https://biconsult.ru/img/datascience-ml-ai/student-mat.csv")
print(data.info())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
predictions = linear_regression.predict(x_test)

# Calculation of R2 Score
from sklearn.model_selection import cross_val_score
print(cross_val_score(linear_regression, x, y, cv=10, scoring="r2").mean())
print(cross_val_score(linear_regression, x, y, cv=10, scoring="r2").max())
print(cross_val_score(linear_regression, x, y, cv=10, scoring="r2").min())

from sklearn.model_selection import GridSearchCV

# Определите сетку гиперпараметров для настройки
param_grid = {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}

# Создайте объект модели (например, ElasticNet, если вы хотите настроить alpha и l1_ratio)
model = ElasticNet()

# Создайте объект GridSearchCV для поиска по сетке
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2')

# Выполните поиск по сетке
grid_search.fit(x, y)

# Получите лучшие гиперпараметры
best_params = grid_search.best_params_
print("Лучшие гиперпараметры:", best_params)