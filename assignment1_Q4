import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

# Initiating import parameters:
m = 50  # [kg/m]     - Mass per unit length
S = 5  # [kgm/m]    - Static mass moment of the wing around x_f
I_alpha = 4.67  # [kgm^2/m]  - Mass moment of the wing around x_f
S_beta = 1.56  # [kgm/m]    - Static mass moment of the control surface around x_h
I_beta = 0.26  # [kgm^2/m]  - Mass moment of the control surface around x_h
I_alpha_beta = 0.81  # [kgm^2/m]  - Product of interia of the control surface

c = 1  # [m]        - Chord
b = c / 2  # [m]        - Semi-chord
a = -0.2  # [-]        - Elastic axis position
ch = 0.5  # [-]        - Hinge axis position

K_h = 25e3  # [N/m]      - Linear heave stiffness
K_alpha = 9e3  # [Nm/rad]   - Linear pitch stiffness
K_beta = 1e3  # [Nm/rad]   - Linear control stiffness
K_h5 = 750 * K_h  # [N/m]      - Non-linear heave stiffness

C_h = K_h / 1000  # [N/m]      - Structual damping
C_alpha = K_alpha / 1000  # [N/m]      - Structual damping
C_beta = K_beta / 1000  # [N/m]      - Structual damping
rho = 1.225  # [kg/m^3]   - Air density

# Wagner constants:
psi_1 = 0.165
psi_2 = 0.335
epsi_1 = 0.0455
epsi_2 = 0.3

# Theodorsen constants:
mu = np.arccos(ch)

T1 = -1 / 3 * np.sqrt(1 - ch ** 2) * (2 + ch ** 2) + ch * mu
T2 = ch * (1 - ch ** 2) - np.sqrt(1 - ch ** 2) * (1 + ch ** 2) * mu + ch * (mu) ** 2
T3 = -(1 / 8 + ch ** 2) * mu ** 2 + 1 / 4 * ch * np.sqrt(1 - ch ** 2) * mu * (7 + 2 * ch ** 2) - 1 / 8 * (
            1 - ch ** 2) * (5 * (ch ** 2) + 4)

T4 = -mu + ch * np.sqrt(1 - ch ** 2)
T5 = -(1 - ch ** 2) - (mu ** 2) + 2 * ch * np.sqrt(1 - ch ** 2) * mu
T6 = T2

T7 = -(1 / 8 + ch ** 2) * mu + 1 / 8 * ch * np.sqrt(1 - ch ** 2) * (7 + 2 * ch ** 2)
T8 = -1 / 3 * np.sqrt(1 - ch ** 2) * (2 * ch ** 2 + 1) + ch * mu
T9 = 1 / 2 * (1 / 3 * ((np.sqrt(1 - ch ** 2)) ** 3) + a * T4)

T10 = np.sqrt(1 - ch ** 2) + mu
T11 = mu * (1 - 2 * ch) + np.sqrt(1 - ch ** 2) * (2 - ch)
T12 = np.sqrt(1 - ch ** 2) * (2 + ch) - mu * (2 * ch + 1)

T13 = 1 / 2 * (-T7 - (ch - a) * T1)
T14 = 1 / 16 + 1 / 2 * a * ch


def matrices(rho, U, amplitude, K_h_new):
    # Structual mass matrix [A}:
    A = np.array([[m, S, S_beta],
                  [S, I_alpha, I_alpha_beta],
                  [S_beta, I_alpha_beta, I_beta]])

    # Aerodynamic mass matrix [B]:
    B = b ** 2 * np.array([[np.pi, -np.pi * a * b, -T1 * b],
                           [-np.pi * a * b, np.pi * (b ** 2) * (1 / 8 + a ** 2), -(T7 + (ch - a) * T1) * b ** 2],
                           [-T1 * b, 2 * T13 * b ** 2, (-T3 * b ** 2) / np.pi]])

    # Mass matrix [M]:
    M = A + rho * B

    C_h=K_h_new/1000
    # Structual Damping [C]:
    C = np.array([[C_h, 0, 0],
                  [0, C_alpha, 0],
                  [0, 0, C_beta]])

    # Aerodynamic Damping [D]:
    D1 = b ** 2 * np.array([[0, np.pi, -T4],
                            [0, np.pi * (0.5 - a) * b, (T1 - T8 - (ch - a) * T4 + 0.5 * T11) * b],
                            [0, (-2 * T9 - T1 + T4 * (a - 0.5)) * b, (-T4 * T11 * b) / (2 * np.pi)]])

    D2 = np.array([[2 * np.pi * b, 2 * np.pi * (b ** 2) * (0.5 - a), b ** 2 * T11],
                   [-2 * np.pi * (b ** 2) * (a + 0.5), -2 * np.pi * (b ** 3) * (a + 0.5) * (0.5 - a),
                    -(b ** 3) * (a + 0.5) * T11],
                   [(b ** 2) * T12, (b ** 3) * T12 * (0.5 - a), (b ** 3 * T12 * T11) / (2 * np.pi)]])

    D = D1 + (1 - psi_1 - psi_2) * D2

    # Structual stiffness matrix [E]:
    E = np.array([[K_h, 0, 0],
                  [0, K_alpha, 0],
                  [0, 0, K_beta]])

    # Structual stiffness matrix [E]:
    E_eqv = np.array([[K_h + (5 / 8) * (amplitude ** 4) * K_h5, 0, 0],
                      [0, K_alpha, 0],
                      [0, 0, K_beta]])

    # Aerodynamic stiffness matrix [F]:
    F1 = b ** 2 * np.array([[0, 0, 0],
                            [0, 0, (T4 + T10)],
                            [0, 0, (T5 - T4 * T10) / np.pi]])

    F2 = np.array([[0, 2 * np.pi * b, 2 * b * T10],
                   [0, -2 * np.pi * (b ** 2) * (a + 0.5), -2 * (b ** 2) * (a + 0.5) * T10],
                   [0, (b ** 2) * T12, ((b ** 2) * T12 * T10) / np.pi]])

    F3 = np.array([[2 * np.pi * b, 2 * np.pi * (b ** 2) * (0.5 - a), (b ** 2) * T11],
                   [-2 * np.pi * (b ** 2) * (a + 0.5), -2 * np.pi * (b ** 3) * (a + 0.5) * (0.5 - a),
                    -(b ** 3) * (a + 0.5) * T11],
                   [(b ** 2) * T12, (b ** 3) * T12 * (1 / 2 - a), ((b ** 3) * T12 * T11) / (2 * np.pi)]])

    F = F1 + (1 - psi_1 - psi_2) * F2 + (((psi_1 * epsi_1) / b) + ((psi_2 * epsi_2) / b)) * F3

    # Aerodynamic influence matrix [W]:
    W0 = np.array([[-psi_1 * (epsi_1 / b) ** 2],
                   [-psi_2 * (epsi_2 / b) ** 2],
                   [psi_1 * epsi_1 * (1 - epsi_1 * (0.5 - a)) / b],
                   [psi_2 * epsi_2 * (1 - epsi_2 * (0.5 - a)) / b],
                   [psi_1 * epsi_1 * (T10 - ((epsi_1 * T11) / 2)) / (np.pi * b)],
                   [psi_2 * epsi_2 * (T10 - ((epsi_2 * T11) / 2)) / (np.pi * b)]])

    # W = np.array([[2 * np.pi * b * W0.T],
    #              [-2 * np.pi * (b ** 2) * (a + 0.5) * W0.T],
    #              [b ** 2 * T12 * W0.T]])
    W = np.array([[2 * np.pi * b * W0.T, -2 * np.pi * (b ** 2) * (a + 0.5) * W0.T,
                   b ** 2 * T12 * W0.T]])  # minus infront of the 2??
    W = W.reshape([3, 6])

    # Aerodynamic state equation matrices [W1] and [W2]:
    W1 = np.array([[1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1]])

    W2 = np.array([[-epsi_1 / b, 0, 0, 0, 0, 0],
                   [0, -epsi_2 / b, 0, 0, 0, 0],
                   [0, 0, -epsi_1 / b, 0, 0, 0],
                   [0, 0, 0, -epsi_2 / b, 0, 0],
                   [0, 0, 0, 0, -epsi_1 / b, 0],
                   [0, 0, 0, 0, 0, -epsi_2 / b]])

    # Taking the inverse of [M]:
    M_inv = np.linalg.inv(M)

    Q_eqv = np.block([[-M_inv @ (C + rho * U * D), -M_inv @ (E_eqv + rho * (U ** 2) * F), -rho * (U ** 3) * M_inv @ W],
                      [np.eye(3), np.zeros(shape=(3, 3)), np.zeros(shape=(3, 6))],
                      [np.zeros(shape=(6, 3)), W1, U * W2]])

    return Q_eqv


# def stability_assesment(U, amp):
#     delta_A = 0.01
#     # outside
#     Q_eqv = matrices(1.225, U, amp + delta_A)
#     Q_eqv_eigenvals, _ = np.linalg.eig(Q_eqv)
#     real_parts = np.real(Q_eqv_eigenvals)
#     if np.all(real_parts < 0):
#         stb_out = 1  # Stable from the outside
#     else:
#         stb_out = 0  # Unstable from the outside
#
#     # inside
#     Q_eqv = matrices(1.225, U, amp - delta_A)
#     Q_eqv_eigenvals, _ = np.linalg.eig(Q_eqv)
#     real_parts = np.real(Q_eqv_eigenvals)
#     if np.all(real_parts < 0):
#         stb_in = 0  # Unstable from the outside
#     else:
#         stb_in = 1  # Stable from the outside
#
#     return stb_out, stb_in


## NEW CODE

omega_h_list = []
omega_alpha_list = []
omega_beta_list = []
damping_h_list = []
damping_alpha_list = []
damping_beta_list = []
U_list = []

eig_real_part_list1 = []
eig_real_part_list2 = []
eig_real_part_list3 = []

eig_imag_part_list1 = []
eig_imag_part_list2 = []
eig_imag_part_list3 = []
eig_imag_part_list4 = []
eig_imag_part_list5 = []
eig_imag_part_list6 = []

Uf = []
Uf_stable = []
Uf_unstable = []
Uf_hstable = []

A_stable = []
A_unstable = []
A_hstable = []
stable=[]
unstable=[]
counter=0
# for amp in np.arange(0, 1, 0.005):
#     flut = 0
#     U = 5
#     # print(amp)
#     while flut == 0:
#         Q_eqv = matrices(1.225, U, amp)
#
#         V = []
#         # determine fixed point
#         Q_eqv_eigenvals, Q_eqv_eigenvec = np.linalg.eig(Q_eqv)
#         # print('hello')
#         # print(Q_eqv_eigenvec)
#
#
#         # L = np.diag(Q_eqv_eigenvals)
#         # V = Q_eqv_eigenvec
#
#         omega_h = np.sqrt(np.real(Q_eqv_eigenvals[0]) ** 2 + np.imag(Q_eqv_eigenvals[0]) ** 2)
#         omega_alpha = np.sqrt(np.real(Q_eqv_eigenvals[2]) ** 2 + np.imag(Q_eqv_eigenvals[2]) ** 2)
#         omega_beta = np.sqrt(np.real(Q_eqv_eigenvals[4]) ** 2 + np.imag(Q_eqv_eigenvals[4]) ** 2)
#         omega_h_list.append(omega_h)
#         omega_alpha_list.append(omega_alpha)
#         omega_beta_list.append(omega_beta)
#
#         # Compute dampings
#         damping_h = -np.real(Q_eqv_eigenvals[0]) / omega_h
#         damping_alpha = -np.real(Q_eqv_eigenvals[2]) / omega_alpha
#         damping_beta = -np.real(Q_eqv_eigenvals[4]) / omega_beta
#
#         if damping_h < 0 or damping_alpha < 0 or damping_beta < 0:
#             Uf.append(U)
#             stb_out, stb_in = stability_assesment(U, amp)
#
#             if stb_out == 1 and stb_in == 1:
#                 Uf_stable.append(U)
#                 A_stable.append(amp)
#             elif stb_out == 0 and stb_in == 0:
#                 Uf_unstable.append(U)
#                 A_unstable.append(amp)
#             else:
#                 Uf_hstable.append(U)
#                 A_hstable.append(amp)
#             flut = 1
#
#         U = U + 0.1  # 0.01

K_h_list=np.linspace(1e3, 1000e3, 100)
for i in range(len(K_h_list)):
    Q_eqv_new=matrices(1.225, 58.4, 0, K_h_list[i])
    Q_eqv_eigenvals_new, Q_eqv_eigenvec_new = np.linalg.eig(Q_eqv_new)

    omega_h = np.sqrt(np.real(Q_eqv_eigenvals_new[0]) ** 2 + np.imag(Q_eqv_eigenvals_new[0]) ** 2)
    omega_alpha = np.sqrt(np.real(Q_eqv_eigenvals_new[2]) ** 2 + np.imag(Q_eqv_eigenvals_new[2]) ** 2)
    omega_beta = np.sqrt(np.real(Q_eqv_eigenvals_new[4]) ** 2 + np.imag(Q_eqv_eigenvals_new[4]) ** 2)
    omega_h_list.append(omega_h/(2*np.pi))
    omega_alpha_list.append(omega_alpha/(2*np.pi))
    omega_beta_list.append(omega_beta/(2*np.pi))

    # Compute dampings
    damping_h = -np.real(Q_eqv_eigenvals_new[0]) / omega_h
    damping_alpha = -np.real(Q_eqv_eigenvals_new[2]) / omega_alpha
    damping_beta = -np.real(Q_eqv_eigenvals_new[4]) / omega_beta
    damping_h_list.append(damping_h)
    damping_alpha_list.append(damping_alpha)
    damping_beta_list.append(damping_beta)

plt.figure()
plt.plot(K_h_list, omega_h_list, label='h')
plt.scatter(K_h_list, omega_alpha_list, label='alpha')
plt.scatter(K_h_list, omega_beta_list, label='beta')
plt.xlabel('$K_h$ [m/s]')
plt.ylabel('-')
plt.legend()
plt.show()

plt.figure()
plt.plot(K_h_list, damping_h_list, label='h')
plt.scatter(K_h_list, damping_alpha_list, label='alpha')
plt.scatter(K_h_list, damping_beta_list, label='beta')
plt.xlabel('$K_h$ [m/s]')
plt.ylabel('-')
plt.legend()
plt.show()

# h_dot,alpha_dot, beta_dot, h, alpha, beta, w1, w2, w3, w4, w5, w6=sp.symbols('h_dot alpha_dot beta_dot h  alpha beta w1 w2 w3 w4 w5 w6')
# lambda_=sp.symbols('lambda')
#
# X=sp.Matrix([h_dot,alpha_dot, beta_dot, h, alpha, beta, w1, w2, w3, w4, w5, w6])
#
# F=sp.Matrix([Q_eqv - lambda_*np.identity(12)])
# Jacobian =F.jacobian(X)
#
# print(Jacobian)

# plt.figure()
# plt.scatter(Uf_stable, A_stable, label='stable')
# plt.scatter(Uf_unstable, A_unstable, label='unstable')
# plt.scatter(Uf_hstable, A_hstable, label='half stable')
# plt.xlabel('$U_F$ [m/s]')
# plt.ylabel('A')
# plt.legend()
# plt.show()

# plt.figure()
# plt.scatter(U_list, eig_imag_part_list1)
# plt.scatter(U_list, eig_imag_part_list2)
# plt.scatter(U_list, eig_imag_part_list3)
# plt.scatter(U_list, eig_imag_part_list4)
# plt.scatter(U_list, eig_imag_part_list5)
# plt.scatter(U_list, eig_imag_part_list6)
# plt.title('Imag part eigenvalues')
# plt.grid()
# plt.show()

# plt.figure()
# plt.scatter(U_list, eig_real_part_list1)
# plt.scatter(U_list, eig_real_part_list2)
# plt.scatter(U_list, eig_real_part_list3)
# plt.title('Real part eigenvalues')
# plt.grid()
# plt.show()


