import numpy as np
import pandas as pd

# Загрузка данных
dataset = pd.read_csv('50_Startups.csv')

# One-Hot Encoding категориальной переменной 'State'
dataset = pd.get_dummies(dataset, columns=['State'])

# Преобразование значений из bool в int
for column in dataset.columns[-3:]:
    dataset[column] = dataset[column].astype(int)

# Вывод первых строк данных для проверки
print(dataset.head())

# Отделение признаков и целевой переменной
X = dataset.drop(columns=['Profit']).values
y = dataset['Profit'].values

# Деление на тренировочный и тестовый наборы
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Обучение модели линейной регрессии
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Предсказание результатов на тестовом наборе
y_pred = regressor.predict(X_test)

# Вывод предсказанных результатов
print(y_pred)

# Оценка модели
import statsmodels.api as sm
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Выполнение отбора признаков
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
