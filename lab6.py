import matplotlib.pyplot as plt
import pandas as pd

# Чтение данных из файла CSV
dataset = pd.read_csv('Salary_Data.csv')

# Проверка первых строк данных
print("Первые строки данных:")
print(dataset.head())

# Определение признаков (X) и целевой переменной (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Вывод первых 5 записей матрицы признаков и зависимой переменной
print("Матрица признаков:")
print(X[:5])
print("Зависимая переменная:")
print(y[:5])

# Разделение данных на обучающий и тестовый наборы
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Обучение модели линейной регрессии
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозирование значений на тестовом наборе
y_pred = regressor.predict(X_test)
print("Прогнозируемые значения:")
print(y_pred)

# Создание областей для графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# График для тренировочных данных
ax1.scatter(X_train, y_train, color='red')
ax1.plot(X_train, regressor.predict(X_train), color='blue')
ax1.set_title('Salary vs Experience (Training set)')
ax1.set_xlabel('Years of Experience')
ax1.set_ylabel('Salary')

# График для тестовых данных
ax2.scatter(X_test, y_test, color='red')
ax2.plot(X_train, regressor.predict(X_train), color='blue')
ax2.set_title('Salary vs Experience (Test set)')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')

# Показать оба графика
plt.show()
