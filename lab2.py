import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

# Ввод переменной t и радиусов необходимых окружностей + ввод угла поворота шариков
t = sp.Symbol('t')
R = 2
phi = 3 * sp.sin(t)

# Построение графика и подграфика с выравниванием осей
fig = plt.figure(figsize=(17, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')

a = np.linspace(0, 2 * math.pi, 500)
w = np.linspace(0, 2 * math.pi, 500)
teta = np.linspace(-1 * math.pi / 2, 0, 500)

conline, = ax1.plot([np.cos(a[0]) * np.sqrt(R+1), 0], [-1, R], 'black')
P, = ax1.plot(np.cos(a[0]) * np.sqrt(R+1), R * np.sin(teta[0]), marker='o', color='black')
Circ, = ax1.plot(R * np.cos(a[0]) * np.cos(w), R * np.sin(w), 'black')
# Доп графики
ax2 = fig.add_subplot(4, 2, 2)
T = np.linspace(0, 2 * math.pi, 1000)
x = sp.sin(t)+2
y = sp.cos(t+math.pi)+2
Vx = sp.diff(x, t)
Vy = sp.diff(y,t)
T = np.linspace(0, 10, 1000)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
for i in np.arange(len(T)):
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
ax2.plot(T, VX)
ax2.set_xlabel('T')
ax2.set_ylabel('VX')
ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, VY)
ax3.set_xlabel('T')
ax3.set_ylabel('VY')

end = 0
def anima(i):
    global end
    if end != 499:
        end = end + 1
    P.set_data(R * np.cos(a[i]) * np.sin(w[end]/4), R * np.sin(teta[end]))
    conline.set_data([R * np.cos(a[i]) * np.sin(w[end]/4), 0], [R * np.sin(teta[end]), R])
    Circ.set_data(R * np.cos(a[i]) * np.cos(w), R * np.sin(w))
    return Circ, P, conline


anim = FuncAnimation(fig, anima, frames=500, interval=1, blit=True)
plt.show()