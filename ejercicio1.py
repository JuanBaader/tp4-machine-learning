import itertools
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import statsmodels.api as sm


def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1', 'variable_2', 'r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)

    return (corr_mat)

def plot_corr_matrix(corr_mat):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        ax=ax
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
    )

    ax.tick_params(labelsize=10)

df = pd.read_excel(r"./resources/acath.xls", engine="xlrd")
df = df.fillna(df.mean())

x = df.loc[:, df.columns != 'sigdz']
y = df.loc[:, df.columns == 'sigdz']

X_train, X_test, Y_train, Y_test = train_test_split(x.loc[:, ['age', 'choleste', 'cad.dur']], y, test_size=0.6, random_state=0)

corr_matrix = df.select_dtypes(include=['float64', 'int']).corr(method='pearson')
print(tidy_corr_matrix(corr_matrix).head(10))
plot_corr_matrix(corr_matrix)

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=Y_train, exog=X_train,)
modelo = modelo.fit()

corr_matrix = df.loc[:, ['age', 'choleste', 'cad.dur', 'sigdz']].corr(method='pearson')
plot_corr_matrix(corr_matrix)
print(tidy_corr_matrix(corr_matrix).head(10))

X_train, X_test, Y_train, Y_test = train_test_split(x.loc[:, ['age', 'choleste', 'cad.dur', 'sex']], y, test_size=0.6, random_state=0)

corr_matrix = df.select_dtypes(include=['float64', 'int']).corr(method='pearson')
print(tidy_corr_matrix(corr_matrix).head(10))
plot_corr_matrix(corr_matrix)

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=Y_train, exog=X_train,)
modelo = modelo.fit()

corr_matrix = df.loc[:, ['age', 'choleste', 'cad.dur', 'sex', 'sigdz']].corr(method='pearson')
plot_corr_matrix(corr_matrix)
print(tidy_corr_matrix(corr_matrix).head(10))

plt.show()