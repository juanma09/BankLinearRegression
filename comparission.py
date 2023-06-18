import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def abline(x, y):
    """Plot a line from slope and intercept"""
    plt.xlabel("Correlation: " + str(pearsonr(x,y)[0]))
    p = np.poly1d(np.polyfit(x, y, 1))
    x_line = np.linspace(np.amin(x), np.amax(x), 200)
    plt.plot(x_line, p(x_line),linestyle='dashed',color="#0A6522")

def title(i):
    if i == 0:
        return "Banco BASA S.A."
    if i == 1:
        return "Banco Continental S.A.E.C.A."
    if i == 2:
        return "Banco Atlas S.A." 

for i in range(0,3):
    slope = 0
    data = pd.read_excel("Excel/output.xlsx", sheet_name=i)
    data = data.dropna()
    df = pd.DataFrame(data)
    df['TIME'] = df['FECHA'].dt.month + 12*(df['FECHA'].dt.year-2018)

    x = df['TIME']
    y = df[['Inversiones en valores', 'DEPOSITOS', 'COLOCACIONES NETAS', 'ROA']]

    print(title(i), "\n")
    #plt.title(title(i))
    plt.scatter(x,y['Inversiones en valores'],color='#CB3204')
    plt.ylabel('INVESTMENTS')
    abline(x,y['Inversiones en valores'])
    slope += pearsonr(x,y['Inversiones en valores'])[0]
    plt.show()

    #plt.title(title(i))
    plt.scatter(x,y['DEPOSITOS'],color='#CB3204')
    plt.ylabel('DEPOSITS')
    abline(x,y['DEPOSITOS'])
    slope += pearsonr(x,y['Inversiones en valores'])[0]
    plt.show()

    plt.scatter(x,y['COLOCACIONES NETAS'],color='#CB3204')
    plt.ylabel('LOANS')
    abline(x,y['COLOCACIONES NETAS'])
    slope += pearsonr(x,y['Inversiones en valores'])[0]
    plt.show()

    plt.scatter(x,y['ROA'],color='#CB3204')
    plt.ylabel('ROA')
    abline(x,y['ROA'])
    slope += pearsonr(x,y['Inversiones en valores'])[0]
    plt.show()

    slope /= 4
    print("SLOPE: ", slope)