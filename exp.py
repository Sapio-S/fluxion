import numpy as np

def func(x):
    return [i*10 for i in x]

dx = np.random.normal(1, 0.1, 100)
dy = np.random.normal(2, 0.1, 100)

x = np.arange(100,200,1)
np.random.shuffle(x)
y = func(x) # groundtruth

realx = x+dx
realy = func(realx) + dy

# get expected error
errs = []
for i in range(100):
    errs.append(realy[i] - y[i])
print(np.mean(errs))

