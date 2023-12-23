import re

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import openpyxl
data = pd.read_excel('poriv.xlsx', engine='openpyxl')
# print(data.info())
# print(data.isna().sum())
target = data['Ki (действ)']
target = target.drop(55)
print(target.shape[0])

def remove_and_convert(cell_value):
    if isinstance(cell_value, str):
        # Используем регулярное выражение для поиска текста в круглых скобках
        pattern = r'\((.*?)\)'
        matches = re.findall(pattern, cell_value)
        if matches:
            for match in matches:
                cell_value = cell_value.replace(f'({match})', '').strip()

        # Заменяем запятую на точку, если она присутствует
        cell_value = cell_value.replace(',', '.')
        cell_value = cell_value.split(' ')[0]
        print(cell_value)

        try:
            value = float(cell_value)  # Пробуем преобразовать во float
            return value
        except ValueError:
            pass  # Если не удалось преобразовать в float, продолжаем
    return cell_value
data[['Протяженность заменямого участка, м', 'обратка']] = data['Протяженность заменямого участка, м'].str.split(';', expand=True)

data['Протяженность заменямого участка, м'] = data['Протяженность заменямого участка, м'].apply(remove_and_convert)
print(data['Протяженность заменямого участка, м'].dtype)
print(111)
def convert_year_to_date(year):
    if isinstance(year, str) and year.startswith('до '):
        year = year.replace('до ', '1/1/')
    return pd.to_datetime(year, errors='coerce')
data['Год ввода в эксплуатацию'] = data['Год ввода в эксплуатацию'].apply(convert_year_to_date)
data['Наличие других порывов на участке, К2'] = data['Наличие других порывов на участке, К2'].replace({'да': True, 'нет': False}).astype(bool)
# data['Коррозионная активность грунта, К3'] = data['Коррозионная активность грунта, К3'].factorize()[0]
data['Наличие/отсутсвие затопления (следов затопления) канала, К4'] = data['Наличие/отсутсвие затопления (следов затопления) канала, К4'].replace({'да': True, 'нет': False}).astype(bool)
data['Наличие пересечений с коммуникациями, К5'] = data['Наличие пересечений с коммуникациями, К5'].replace({'да': True, 'нет': False}).astype(bool)

data['Коррозионная активность грунта, К3'], categories = data['Коррозионная активность грунта, К3'].factorize()
print(data['Коррозионная активность грунта, К3'])
for code, category in enumerate(categories):
    print(f'Код: {code}, Значение: {category}')

data['Материал трубопровода'], categories1 = data['Материал трубопровода'].factorize()
print(data['Материал трубопровода'])
for code, category in enumerate(categories1):
    print(f'Код: {code}, Значение: {category}')
print(data['Материал трубопровода'].dtype)

data['Тип прокладки'], categories2 = data['Тип прокладки'].factorize()
print(data['Тип прокладки'])
for code, category in enumerate(categories2):
    print(f'Код: {code}, Значение: {category}')
print(data['Тип прокладки'].dtype)
# data['Продолжительность устранения порыва'] = pd.to_datetime(data['Продолжительность устранения порыва'])
# data['Продолжительность устранения порыва'] = data['Продолжительность устранения порыва'].hour
data['Продолжительность устранения порыва'] = data['Продолжительность устранения порыва'].astype(str)
# Разделяем значения по символу ":"
def convert_time_to_duration(time_value):
    if isinstance(time_value, str):
        try:
            hours, minutes, seconds = map(float, time_value.split(':'))
            if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
                return hours + minutes / 60
            else:
                return None
        except ValueError:
            return None
    elif isinstance(time_value, (int, float)):
        # Если значение уже числовое (часы после минуты), возвращаем его
        return time_value
    else:
        return None
# data[['Часы', 'Минуты', 'Секунды']] = data['Продолжительность устранения порыва'].str.split(':|\.', expand=True)
data['Продолжительность устранения порыва'] = data['Продолжительность устранения порыва'].apply(convert_time_to_duration)
# print(data['Продолжительность устранения порыва'])
# duration = data['Продолжительность устранения порыва'].str.split(':').str
# print(type(duration))
# print(duration)
print(1111111111)
# data[['Часы', 'Минуты', 'Секунды']] = data['Продолжительность устранения порыва'].str.split(':|\.', expand=True)
# Преобразуем значения в числовые типы
# data['Часы'] = pd.to_numeric(data['Часы'])
# data['Минуты'] = pd.to_numeric(data['Минуты'])
# data['Часы'] = data['Продолжительность устранения порыва'].str.extract(r'(\d+)[ч:](\d+)[м]?')
# data['Часы'] = pd.to_numeric(data['Часы'])
# data['Минуты'] = pd.to_numeric(data[0].str.extract(r'(\d+)')[0])
#
# # Создаем столбец с продолжительностью в виде объектов времени
# data['Продолжительность_время'] = pd.to_timedelta(data['Часы'], unit='h') + pd.to_timedelta(data['Минуты'], unit='m')
# # print(data['Продолжительность устранения порыва'].str)
# print(1)
# data['Часы'], data['Минуты'] = data['Продолжительность устранения порыва'].str.split(':').str
#
# # Преобразуем значения в числовые типы
# data['Часы'] = pd.to_numeric(data['Часы'])
# data['Минуты'] = pd.to_numeric(data['Минуты'])
#
# # Создаем столбец с продолжительностью в виде объектов времени
# data['Продолжительность устранения порыва'] = pd.to_timedelta(data['Часы'], unit='h') + pd.to_timedelta(data['Минуты'], unit='m')


data = data.iloc[:, list(range(16)) + [18]]
string_columns = data.select_dtypes(include=['object']).columns.tolist()
print(string_columns)
data.drop(columns=string_columns, axis=1, inplace=True)

missing_values = data.isnull()
data.drop('№ п/п', inplace=True, axis=1)

print(data.describe())



for column in data.columns:
    if missing_values[column].any():
        mean_value = data[column].mean()
        data[column].fillna(mean_value, inplace=True)

missing_values = data.isnull()
data = data.drop(55)


corr = data.corr()
sns.heatmap(corr, annot=True)
# plt.show()
data['Год ввода в эксплуатацию'] = (pd.to_datetime('now') - data['Год ввода в эксплуатацию']).astype('int64')
print(data.info())
print(data.shape[0])
s = data.shape[0]
# data = data.drop(s-1, axis=0)
# df = data.drop(index=range(55, 56, len(data)-1))
# data = data.iloc[:55]
# EDA
print(data.isna().sum())
print(data.info())
print(data.describe())
# Гистограммы для числовых столбцов
numeric_columns = data.select_dtypes(include=['float64', 'int64'])

numeric_columns.hist(figsize=(12, 8))
plt.show()

# Создаем фигуру
fig = plt.figure(figsize=(12, 8))

# Название фигуры
fig.suptitle('Distribution of data across columns')

# Список столбцов (исключая "Date")
columns_to_plot = data.columns.tolist()

# Построение гистограмм для каждого столбца
for i, column in enumerate(columns_to_plot):
    plt.subplot(5, 3, i + 1)
    sns.histplot(data=data, x=column, kde=True)
    plt.title(column)
numeric_columns = data.select_dtypes(include=['float64', 'int64'])


# Регулировка расположения графиков
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
plt.show()
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(features_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Оценка производительности модели на тестовых данных
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')








