import matplotlib.pyplot as plt

def f(x):
    return x*x - x
def f2(x):
    return -x*x + x

trenutno_x = -100
dx = 0.1
y1 = []
y2 = []
x = []
for _ in range(2000):
    x.append(trenutno_x)
    y1.append(f(trenutno_x))
    y2.append(f2(trenutno_x))
    trenutno_x += dx

plt.plot(x,y1,c='red', label='f(x)')
plt.plot(x,y2,c='green', label='f2(x)')
ax = plt.gca()
ax.legend()
plt.show()