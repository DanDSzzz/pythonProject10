import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Загрузка данных и препроцессинг
data = pd.read_csv("https://biconsult.ru/img/datascience-ml-ai/student-mat.csv")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Масштабирование данных (стандартизация)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Создание объекта PCA
pca = PCA()

# Применение PCA к данным
pca.fit(x_scaled)

# Вычисление кумулятивной объясненной дисперсии
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Найдите количество компонентов для сохранения 95% дисперсии
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print("Количество компонентов для сохранения 95% дисперсии:", n_components_95)

# Применение PCA с оптимальным количеством компонентов
pca = PCA(n_components=n_components_95)
x_pca = pca.fit_transform(x_scaled)

# Разделение данных на обучающий и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)

# Создание модели нейросети
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(n_components_95,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Оценка производительности модели
test_loss, test_mae = model.evaluate(x_test, y_test)
print("Mean Absolute Error:", test_mae)