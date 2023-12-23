import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 13, 4, 5, 6],
    'Feature3': [3, 4, 24, 5, 6],
    'Feature4': [6, 2, 23, 14, 5],
    'Feature5': [7, 93, 34, 55, 26],
    'Feature6': [9, 14, 41, 25, 46],
    'Target': [3, 4, 5, 6, 7]
})

X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6']]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=[['PCA1', 'PCA2', 'PCA3']])

X_train, X_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.5, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mse:', mse)
print('R2:', r2)