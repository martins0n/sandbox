import pandas as pd
import math
import numpy as np
import scipy.optimize as opt
df = pd.read_csv('data.csv', header=None, names=['x', 'y'])

Y = df['y'].values
X = df['x'].values
print(df.head())

def func(x):
    a, b, c = x[0], x[1], x[2]
    return ((Y - (a * np.sin(X) + b * np.log(X)) ** 2 - c * X ** 2) ** 2).sum()



a = opt.minimize(func, [1, 1, 1])

print(a.x)

with open('answer.txt', 'w') as f:
    f.write(' '.join(map(lambda x: f"{x:.2f}", a.x)))
