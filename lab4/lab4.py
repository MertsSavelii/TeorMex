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

def formY2(y, t, fOm):
    y1,y2 = y
    dydt = [y2,fOm(y1,y2)]
    return dydt

# defining parameters
# the angle of the plane (and the prism)
alpha = 0.002
M = 1
m = 0.1
R = 0.3
c = 20
l0 = 0.7
g = 9.81

# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

# defining s, phi, V=ds/dt and om=dphi/dt as functions of 't'
phi=0
psi=sp.Function('psi')(t)
Vphi=0
Vpsi=sp.Function('Vpsi')(t)

l = 2 * R * sp.cos(psi)  # длина пружины
#constructing the Lagrange equations
#1 defining the kinetic energy
TT1 = M * R**2 * Vphi**2 / 4
V1 = 2*Vpsi * R
V2 = Vphi * R * sp.sin(2 * psi)
Vr2 = V1**2 + V2**2
TT2 = m * Vr2 / 2
TT = TT1+TT2
# 2 defining the potential energy
Pi1 = 2 * R * m * g * sp.sin(psi)**2
Pi2 = (c * (l - l0)**2) / 2
Pi = Pi1+Pi2
# 3 Not potential force
M = alpha * phi**2;

# Lagrange function
L = TT-Pi

# тут исследую положения устойчивости при л0 = 0.3
PI = -1*R*sp.cos(2*psi)*m*g + (c/2)*(2*R*sp.cos(psi)-l0)*(2*R*sp.cos(psi)-l0)
print(sp.diff(PI,t))
PI1 = -6.0*(0.6*sp.cos(psi) - 0.502)*sp.sin(psi) - 0.6*(6.0*sp.cos(psi) - 0.502)*sp.sin(psi) + 0.5886*sp.sin(2*psi)#первая производная ПИ
print(sp.diff(PI1,t))
PI2 = (0.3012 - 3.6*sp.cos(0))*sp.cos(0) + (3.012 - 3.6*sp.cos(0))*sp.cos(0) + 7.2*sp.sin(0)**2 + 1.1772*sp.cos(2*0)# вторая производная ПИ
print(PI2)

# тут уже проверка устойчивости при пси = 0 и л = 0.7
print(sp.diff(sp.diff(PI, t), t))
pi2 = (4.2 - 3.6*sp.cos(0))*sp.sin(0) + (4.2 - 3.6*sp.cos(0))*sp.cos(0) + (4.2 - 3.6*sp.cos(0))*sp.sin(0) + (4.2 - 3.6*sp.cos(0))*sp.cos(0) + 7.2*sp.sin(0)**2 + 0.5886*sp.sin(2*0) + 1.1772*sp.cos(2*0)
print(pi2)
# equations
#ur1 = sp.diff(sp.diff(L,Vphi),t)-sp.diff(L,phi) - M
ur2 = sp.diff(sp.diff(L,Vpsi),t)-sp.diff(L,psi)

# isolating second derivatives(dV/dt and dom/dt) using Kramer's method
# a11 = ur1.coeff(sp.diff(Vphi,t),1)
# a12 = ur1.coeff(sp.diff(Vpsi,t),1)
# a21 = ur2.coeff(sp.diff(Vphi,t),1)
a22 = ur2.coeff(sp.diff(Vpsi,t),1)
#b1 = -(ur1.coeff(sp.diff(Vphi,t),0)).coeff(sp.diff(Vpsi,t),0).subs([(sp.diff(phi,t),Vphi), (sp.diff(psi,t), Vpsi)])
b2 = -ur2.coeff(sp.diff(Vpsi,t),0).subs(sp.diff(phi,t), Vpsi);
# detA = a11*a22-a12*a21
# detA1 = b1*a22-b2*a21
# detA2 = a11*b2-b1*a21
#
# dVdt = detA1/detA
domdt = b2/a22

countOfFrames = 1700

# Constructing the system of differential equations
T = np.linspace(0, 25, countOfFrames)
# Pay attention here, the function lambdify translate function from the sympy to numpy and then form arrays much more
# faster then we did using subs in previous lessons!
#fVphi = sp.lambdify([phi,psi,Vphi,Vpsi], dVdt, "numpy")
fVpsi = sp.lambdify([psi,Vpsi], domdt, "numpy")
y0 = [np.pi/6, 0]
sol = odeint(formY2, y0, T, args = (fVpsi,))

#sol - our solution
#sol[:,0] - phi
#sol[:,1] - psi
#sol[:,2] - dphi/dt
#sol[:,3] - dpsi/dt

# Построение графика и подграфика с выравниванием осей
fig = plt.figure(figsize=(17, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')

phi = 0
psi = sol[:,0]
# Vphi = sol[:,2]
Vpsi = sol[:,1]

w = np.linspace(0, 2 * math.pi, countOfFrames)
conline, = ax1.plot([sp.sin(2*psi[0]) * R * sp.cos(phi), 0], [-sp.cos(2*psi[0]) * R, R], 'black')
P, = ax1.plot(sp.sin(2*psi[0]) * R * sp.cos(phi), -sp.cos(2*psi[0]) * R, marker='o', color='black')
Circ, = ax1.plot(R * sp.cos(phi) * np.cos(w), R * np.sin(w), 'black')

#Доп графики
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, psi)
ax2.set_xlabel('T')
ax2.set_ylabel('psi')
ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, Vpsi)
ax3.set_xlabel('T')
ax3.set_ylabel('Vpsi')

def anima(i):
    P.set_data(sp.sin(2*psi[i]) * R * sp.cos(phi), -sp.cos(2*psi[i]) * R)
    conline.set_data([sp.sin(2*psi[i]) * R * sp.cos(phi), 0], [-sp.cos(2*psi[i]) * R, R])
    Circ.set_data(R * sp.cos(phi) * np.cos(w), R * np.sin(w))
    return Circ, P, conline

anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=1, blit=True)
plt.show()