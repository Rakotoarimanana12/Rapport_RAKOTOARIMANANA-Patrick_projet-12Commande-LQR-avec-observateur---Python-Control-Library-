import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. MATRICES DU SYSTEME

A = np.array([[0, 1],
              [-2, -3]])

B = np.array([[0],
              [1]])

C = np.array([[1, 0]])

D = np.array([[0]])

# 2. COMMANDE LQR

Q = np.array([[10, 0],
              [0, 1]])

R = np.array([[1]])

K, S, E = ctrl.lqr(A, B, Q, R)

print("Gain LQR K =")
print(K)

# 3. OBSERVATEUR

# pôles observateur plus rapides
poles_obs = [-5, -6]

L = ctrl.place(A.T, C.T, poles_obs).T

print("Gain Observateur L =")
print(L)

# 4. SYSTEME COMPLET
# x = état réel
# xhat = état estimé

def system(t, z):
    x = z[0:2].reshape((2,1))
    xhat = z[2:4].reshape((2,1))

    # sortie réelle
    y = C @ x

    # commande basée sur état estimé
    u = -K @ xhat

    # dynamique réelle
    dx = A @ x + B @ u

    # dynamique observateur
    dxhat = A @ xhat + B @ u + L @ (y - C @ xhat)

    return np.concatenate((dx.flatten(), dxhat.flatten()))

# 5. CONDITIONS INITIALES

x0 = np.array([1, 0])       # état réel initial
xhat0 = np.array([0, 0])    # estimation initiale fausse

z0 = np.concatenate((x0, xhat0))

# 6. SIMULATION

t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)

sol = solve_ivp(system, t_span, z0, t_eval=t_eval)

# 7. EXTRACTION

t = sol.t

x1 = sol.y[0]
x2 = sol.y[1]

x1hat = sol.y[2]
x2hat = sol.y[3]

# 8. AFFICHAGE

plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.plot(t, x1, label="x1 réel")
plt.plot(t, x1hat, '--', label="x1 estimé")
plt.title("Position")
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, x2, label="x2 réel")
plt.plot(t, x2hat, '--', label="x2 estimé")
plt.title("Vitesse")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()