import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def abline(x, y):
    """Plot a line from slope and intercept"""
    p = np.poly1d(np.polyfit(x, y, 1))
    x_line = np.linspace(np.amin(x), np.amax(x), 200)
    plt.plot(x_line, p(x_line), color='g', linestyle='dashed')

def title(i):
    if i == 0:
        return "Banco BASA S.A."
    if i == 1:
        return "Banco Continental S.A.E.C.A."
    if i == 2:
        return "Banco Atlas S.A." 

for i in range(0,3):
    data = pd.read_excel("Excel/output.xlsx", sheet_name=i)
    data = data.dropna()
    df = pd.DataFrame(data)
    df['MONTH'] = df['FECHA'].dt.month

    x = df[['Inversiones en valores','Productos financieros','COLOCACIONES NETAS', 'Cart. Venc. / Cart. Total - Morosidad', 'DEPOSITOS']]
    y = df['ROA']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.06)

    #with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    y_pred = regr.predict(x_test)
    err_per = []
    for i, test in enumerate(list(y_test)):
        err_per.append(100*abs((y_pred[i] - test)/test))
    err_df = pd.DataFrame(err_per)


    print("Prediction: ", y_pred)
    print("Actual values: ", list(y_test))
    print("% of error", err_per, err_df.describe())

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    plt.title(title(i))
    plt.scatter(x['Inversiones en valores'], y, color='#CB3204')
    abline(x['Inversiones en valores'], y)
    plt.xlabel('Inversiones en valores')
    plt.show()

    plt.title(title(i))
    plt.scatter(x['COLOCACIONES NETAS'], y, color='#CB3204')
    abline(x['COLOCACIONES NETAS'],y)
    plt.xlabel('COLOCACIONES NETAS')
    plt.show()

    plt.title(title(i))
    plt.scatter(x['Productos financieros'], y, color='#CB3204')
    abline(x['Productos financieros'], y)
    plt.xlabel('Productos financieros')
    plt.show()

    plt.title(title(i))
    plt.scatter(x['Cart. Venc. / Cart. Total - Morosidad'], y, color='#CB3204')
    abline(x['Cart. Venc. / Cart. Total - Morosidad'], y)
    plt.xlabel('Cart. Venc. / Cart. Total - Morosidad')
    plt.show()

    plt.title(title(i))
    plt.scatter(x['DEPOSITOS'], y, color='#CB3204')
    abline(x['DEPOSITOS'], y)
    plt.xlabel('DEPOSITOS')
    plt.show()


