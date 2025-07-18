import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
pd.set_option("display.max_columns", None)

df_dsa = pd.read_csv("dataset.csv")
# print(df_dsa.shape)
# print(df_dsa.columns)
# print(df_dsa.head())
# print(df_dsa.info())
# print(df_dsa.isnull().sum())
# print(df_dsa.corr())
# print(df_dsa.describe())
# print(df_dsa["horas_estudo_mes"].describe())
# sns.histplot(data = df_dsa, x = "horas_estudo_mes", kde = True)
# plt.show()

X = np.array(df_dsa["horas_estudo_mes"])
# print(type(X))
X = X.reshape(-1, 1)
y = df_dsa["salario"]
# plt.scatter(X, y, color = "blue", label = "Dados Reais Históricos")
# plt.xlabel("Horas de estudo")
# plt.ylabel("Salário")
# plt.legend()
# plt.show()

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42)
# print(X_treino.shape)
# print(X_teste.shape)
# print(y_treino.shape)
# print(y_teste.shape)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)
# plt.scatter(X, y, color = "blue", label = "Dados Reais Históricos")
# plt.plot(X, modelo.predict(X), color = "red", label = "Reta de Regressão com as Previsões do Modelo")
# plt.xlabel("Horas de Estudo")
# plt.ylabel("Salário")
# plt.legend()
# plt.show()
score = modelo.score(X_teste, y_teste)
# print(f"Coeficiente R^2: {score: .2f}")
# print(modelo.intercept_)
# print(modelo.coef_)

horas_estudo_novo = np.array([[48]])
salario_previsto = modelo.predict(horas_estudo_novo)
print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês, seu salário pode ser igual a", salario_previsto)

salario = modelo.intercept_ + (modelo.coef_ * horas_estudo_novo)
print(salario)













