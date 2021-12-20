import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import sympy as sp
import math


def formY(y, t, fV, fOm):
    y1,y2,y3,y4 = y
    dydt = [y3,y4,fV(y1,y2,y3,y4),fOm(y1,y2,y3,y4)]
    return dydt

# defining parameters
# the angle of the plane (and the prism)
alpha = math.pi / 6
M = 1
m = 0.1
R = 0.3
c = 20
l0 = 0.2
g = 9.81

# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

# defining s, phi, V=ds/dt and om=dphi/dt as functions of 't'
phi=sp.Function('phi')(t)
psi=sp.Function('psi')(t)
Vphi=sp.Function('Vphi')(t)
Vpsi=sp.Function('Vpsi')(t)

l = 2 * R * sp.cos(phi)  # длина пружины
#constructing the Lagrange equations
#1 defining the kinetic energy
TT1 = M * R**2 * Vphi**2 / 2
V1 = 2*Vpsi * R
V2 = Vphi * R * sp.sin(2 * psi)
Vr2 = V1**2 + V2**2
TT2 = m * Vr2 / 2
TT = TT1+TT2
# 2 defining the potential energy
Pi1 = m * g *  sp.cos(psi) * l
Pi2 = (c * (l - l0)**2) / 2
Pi = Pi1+Pi2
# 3 Not potential force
M = alpha * phi**2;

# Lagrange function
L = TT-Pi

# equations
ur1 = ((M / 2) + m * sp.sin(2 * psi)**2) * R**2 * sp.diff(Vphi) + 2 * m * R**2 * Vphi*Vpsi*sp.sin(4*psi) - alpha * phi**2
ur2 = m * R * (4 * sp.diff(Vpsi) - Vphi**2 * sp.sin(4*phi)) + 2 * m * g * sp.sin(2*psi) - 2*c*(2*R*sp.cos(psi) - l0)*sp.sin(psi)

# isolating second derivatives(dV/dt and dom/dt) using Kramer's method
a11 = ur1.coeff(sp.diff(Vphi,t),1)
a12 = ur1.coeff(sp.diff(Vpsi,t),1)
a21 = ur2.coeff(sp.diff(Vphi,t),1)
a22 = ur2.coeff(sp.diff(Vpsi,t),1)
b1 = -(ur1.coeff(sp.diff(Vphi,t),0)).coeff(sp.diff(Vpsi,t),0).subs([(sp.diff(phi,t),Vphi), (sp.diff(psi,t), Vpsi)])
b2 = -(ur2.coeff(sp.diff(Vphi,t),0)).coeff(sp.diff(Vpsi,t),0).subs([(sp.diff(phi,t),Vphi), (sp.diff(psi,t), Vpsi)])

detA = a11*a22-a12*a21
detA1 = b1*a22-b2*a21
detA2 = a11*b2-b1*a21

dVdt = detA1/detA
domdt = detA2/detA

countOfFrames = 2000

# Constructing the system of differential equations
T = np.linspace(0, 12, countOfFrames)
# Pay attention here, the function lambdify translate function from the sympy to numpy and then form arrays much more
# faster then we did using subs in previous lessons!
fVphi = sp.lambdify([phi,psi,Vphi,Vpsi], dVdt, "numpy")
fVpsi = sp.lambdify([phi,psi,Vphi,Vpsi], domdt, "numpy")
y0 = [20, -1, -25, 0]
sol = odeint(formY, y0, T, args = (fVphi, fVpsi))

#sol - our solution
#sol[:,0] - phi
#sol[:,1] - psi
#sol[:,2] - dphi/dt
#sol[:,3] - dpsi/dt

print(sol[:,0])

# Ввод переменной t и радиусов необходимых окружностей + ввод угла поворота шариков
t = sp.Symbol('t')
R = 2

# Построение графика и подграфика с выравниванием осей
fig = plt.figure(figsize=(17, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')

# phi = np.linspace(0, 2 * math.pi, 500)
# psi = np.linspace(-math.pi/2, 0, 500)

phi = sol[:,0]
psi = sol[:,1]
w = np.linspace(0, 2 * math.pi, 2000)
conline, = ax1.plot([sp.sin(2*psi[0]) * R * np.abs(sp.cos(phi[0])), 0], [-1, R], 'black')
P, = ax1.plot(sp.sin(2*psi[0]) * R * np.abs(sp.cos(phi[0])), sp.cos(2*psi[0]) * R, marker='o', color='black')
Circ, = ax1.plot(R * np.abs(sp.cos(phi[0])) * np.cos(w), R * np.sin(w), 'black')

#Доп графики
ax2 = fig.add_subplot(4, 2, 2)
T = np.linspace(0, 2 * math.pi, 2000)
x = sp.sin(t)+2
y = sp.cos(t+math.pi)+2
Vx = sp.diff(x, t)
Vy = sp.diff(y,t)
T = np.linspace(0, 10, 2000)
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

def anima(i):
    P.set_data(sp.sin(2*psi[i]) * R * np.abs(sp.cos(phi[i])), sp.cos(2*psi[i]) * R)
    conline.set_data([sp.sin(2*psi[i]) * R * np.abs(sp.cos(phi[i])), 0], [sp.cos(2*psi[i]) * R, R])
    Circ.set_data(R * np.abs(sp.cos(phi[i])) * np.cos(w), R * np.sin(w))
    return Circ, P, conline

anim = FuncAnimation(fig, anima, frames=2000, interval=35, blit=True)
plt.show()