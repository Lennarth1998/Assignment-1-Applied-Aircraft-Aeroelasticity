import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from __future__ import division
import scipy as sp
import scipy.linalg as ln
import scipy.optimize as opt


#Initiating import parameters:
m               = 50    	    #[kg/m]     - Mass per unit length
S               = 5             #[kgm/m]    - Static mass moment of the wing around x_f
I_alpha         = 4.67          #[kgm^2/m]  - Mass moment of the wing around x_f
S_beta          = 1.56          #[kgm/m]    - Static mass moment of the control surface around x_h
I_beta          = 0.26          #[kgm^2/m]  - Mass moment of the control surface around x_h
I_alpha_beta    = 0.81          #[kgm^2/m]  - Product of interia of the control surface 

c               = 1             #[m]        - Chord
b               = c/2           #[m]        - Semi-chord
a               = -0.2          #[-]        - Elastic axis position
ch              = 0.5           #[-]        - Hinge axis position

K_h             = 25e3          #[N/m]      - Linear heave stiffness
K_alpha         = 9e3           #[Nm/rad]   - Linear pitch stiffness
K_beta          = 1e3           #[Nm/rad]   - Linear control stiffness
K_h5            = 750*K_h       #[N/m]      - Non-linear heave stiffness

C_h             = K_h/1000      #[N/m]      - Structual damping
C_alpha         = K_alpha/1000  #[N/m]      - Structual damping
C_beta          = K_beta/1000   #[N/m]      - Structual damping
rho             = 1.225         #[kg/m^3]   - Air density

# Wagner constants:
psi_1           = 0.165
psi_2           = 0.335
epsi_1          = 0.0455
epsi_2          = 0.3

#Theodorsen constants:
mu              = np.arccos(ch)

T1              = -1/3 * np.sqrt(1-ch**2)*(2+ch**2)+ch*mu
T2              = ch*(1-ch**2)-np.sqrt(1-ch**2)*(1+ch**2)*mu+ch*(mu)**2
T3              = -(1/8 + ch**2)*mu**2 + 1/4*ch*np.sqrt(1-ch**2)*mu*(7+2*ch**2)-1/8 * (1-ch**2)*(5 *(ch**2)+4)

T4              = -mu + ch*np.sqrt(1-ch**2)
T5              = -(1-ch**2)-(mu**2) + 2*ch*np.sqrt(1-ch**2)*mu
T6              = T2

T7              = -(1/8+ch**2)*mu + 1/8*ch*np.sqrt(1-ch**2)*(7+2*ch**2)
T8              = -1/3 * np.sqrt(1-ch**2)*(2*ch**2 + 1)+ch*mu
T9              = 1/2 * (1/3*((np.sqrt(1-ch**2))**3)+a*T4)

T10             = np.sqrt(1-ch**2)+mu
T11             = mu*(1-2*ch) + np.sqrt(1-ch**2)*(2-ch)
T12             = np.sqrt(1-ch**2)*(2+ch)-mu*(2*ch + 1)

T13             = 1/2*(-T7 - (ch-a)*T1)
T14             = 1/16 + 1/2 * a*ch

def matrices(rho, U, amplitude):
    # Structual mass matrix [A}:
    A = np.array([[m , S , S_beta],
                 [S , I_alpha, I_alpha_beta],
                 [S_beta,I_alpha_beta, I_beta]])

    # Aerodynamic mass matrix [B]:
    B = b**2 * np.array([[np.pi, -np.pi*a*b, -T1*b],
                         [-np.pi*a*b, np.pi*(b**2)*(1/8 + a**2), -(T7+(ch-a)*T1)*b**2],
                         [-T1*b, 2*T13*b**2, (-T3*b**2)/np.pi]])

    #Mass matrix [M]:
    M = A + rho*B
    
    #Structual Damping [C]:
    C = np.array([[C_h, 0 , 0],
        [0 , C_alpha, 0],
        [0 , 0 , C_beta]])

    # Aerodynamic Damping [D]:
    D1 = b**2 * np.array([[0, np.pi, -T4],
                         [0, np.pi*(0.5-a)*b, (T1-T8-(ch-a)*T4 +0.5*T11)*b],
                         [0, (-2*T9 -T1 + T4*(a-0.5))*b, (-T4*T11*b)/(2*np.pi)]])

    D2 = np.array([[2*np.pi*b,  2*np.pi*(b**2)*(0.5-a), b**2*T11],
                   [-2*np.pi*(b**2)*(a+0.5), -2*np.pi*(b**3)*(a+0.5)*(0.5-a), -(b**3)*(a+0.5)*T11],
                   [(b**2)*T12, (b**3)*T12*(0.5-a), (b**3*T12*T11)/(2*np.pi)]])

    D = D1 + (1-psi_1-psi_2)*D2

    #Structual stiffness matrix [E]:
    E = np.array([[K_h, 0 , 0],
                  [0 , K_alpha, 0],
                  [0 , 0 , K_beta]])


    #Structual stiffness matrix [E]:
    E_eqv = np.array([[K_h+(5/8)*(amplitude**4)*K_h5, 0, 0],
        [0 , K_alpha, 0],
        [0 , 0 , K_beta]])

    #Aerodynamic stiffness matrix [F]:
    F1 = b**2 * np.array([[0,0,0],
                          [0,0,(T4+T10)],
                          [0,0,(T5-T4*T10)/np.pi]])

    F2 = np.array([[0,2*np.pi*b, 2*b*T10],
                   [0,-2*np.pi*(b**2)*(a+0.5),-2*(b**2)*(a+0.5)*T10],
                   [0,(b**2)*T12, ((b**2)*T12*T10)/np.pi]])

    F3 = np.array([[2*np.pi*b, 2*np.pi*(b**2)*(0.5-a), (b**2)*T11],
                   [-2*np.pi*(b**2)*(a+0.5), -2*np.pi*(b**3)*(a+0.5)*(0.5-a), -(b**3)*(a+0.5)*T11],
                   [(b**2)*T12, (b**3)*T12*(1/2-a), ((b**3)*T12*T11)/(2*np.pi)]])

    F= F1+(1-psi_1-psi_2)*F2+ (((psi_1*epsi_1)/b)+((psi_2*epsi_2)/b))*F3

    #Aerodynamic influence matrix [W]:
    W0 = np.array([[-psi_1*(epsi_1/b)**2],
                   [-psi_2*(epsi_2/b)**2],
                   [psi_1*epsi_1*(1-epsi_1*(0.5-a))/b],
                   [psi_2*epsi_2*(1-epsi_2*(0.5-a))/b],
                   [psi_1*epsi_1*(T10-((epsi_1*T11)/2))/(np.pi*b)],
                   [psi_2*epsi_2*(T10-((epsi_2*T11)/2))/(np.pi*b)]])

    #W = np.array([[2 * np.pi * b * W0.T],
    #              [-2 * np.pi * (b ** 2) * (a + 0.5) * W0.T],
    #              [b ** 2 * T12 * W0.T]])
    W = np.array([[2 * np.pi * b * W0.T , 2 * np.pi * (b ** 2) * (a + 0.5) * W0.T, b ** 2 * T12 * W0.T]]) #minus infront of the 2??
    W = W.reshape([3, 6])
    

    #Aerodynamic state equation matrices [W1] and [W2]:
    W1 = np.array([[1,0,0],
                   [1,0,0],
                   [0,1,0],
                   [0,1,0],
                   [0,0,1],
                   [0,0,1]])

    W2 = np.array([[-epsi_1/b,0,0,0,0,0],
                   [0,-epsi_2/b,0,0,0,0],
                   [0,0,-epsi_1/b,0,0,0],
                   [0,0,0,-epsi_2/b,0,0],
                   [0,0,0,0,-epsi_1/b,0],
                   [0,0,0,0,0,-epsi_2/b]])


    #Taking the inverse of [M]:
    M_inv = np.linalg.inv(M)

    Q_eqv = np.block([[-M_inv @ (C+rho*U*D),-M_inv @ (E_eqv+rho*(U**2)*F),-rho*(U**3)*M_inv @ W],
                  [np.eye(3),np.zeros(shape=(3,3)),np.zeros(shape=(3,6))],
                  [np.zeros(shape=(6,3)), W1, U*W2]])

    return Q_eqv

#pd.DataFrame(Q_eqv).to_excel('matrix_Q.xlsx')

### OLD CODE

# U_tab = []
# h_damp = []
# alpha_damp = []
# beta_damp = []
# h_omega = []
# alpha_omega = []
# beta_omega = []
#
#
#
# for U in range(5, 200, 1):
#     Q_eqv = matrices(1.225, U, 0)
#     h_eig = np.linalg.eigvals(Q_eqv)[3]
#     alpha_eig = np.linalg.eigvals(Q_eqv)[4]
#     beta_eig = np.linalg.eigvals(Q_eqv)[5]
#
#     U_tab.append(U)
#     h_omega_value = np.sqrt(h_eig.imag**2 + h_eig.real**2)
#     h_omega.append(h_omega_value)
#     h_damp.append(-(h_eig.real/h_omega_value))
#     alpha_omega_value = np.sqrt(alpha_eig.imag ** 2 + alpha_eig.real ** 2)
#     alpha_omega.append(alpha_omega_value)
#     alpha_damp.append(-(alpha_eig.real / alpha_omega_value))
#     beta_omega_value = np.sqrt(beta_eig.imag ** 2 + beta_eig.real ** 2)
#     beta_omega.append(beta_omega_value)
#     beta_damp.append(-(beta_eig.real / beta_omega_value))
#
#
# plt.figure()
# plt.title("Omega")
# plt.plot(U_tab, h_omega, label='h')
# plt.plot(U_tab, alpha_omega, label=r'$\alpha$', linestyle='dotted')
# plt.plot(U_tab, beta_omega, label=r'$\beta$')
# plt.grid()
# plt.legend()
# #plt.show()
#
# plt.figure()
# plt.title("Damping")
# plt.plot(U_tab, h_damp, label='h')
# plt.plot(U_tab, alpha_damp, label=r'$\alpha$', linestyle='dotted')
# plt.plot(U_tab, beta_damp, label=r'$\beta$')
# plt.grid()
# plt.legend()
# #plt.show()

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

for U in range(5, 200):
    Q_eqv = matrices(1.225, U, 0)

    V = []
    #determine fixed point

    Q_eqv_eigenvals, Q_eqv_eigenvec = np.linalg.eig(Q_eqv)

    #L = np.diag(Q_eqv_eigenvals)
    #V = Q_eqv_eigenvec

    omega_h = np.sqrt(np.real(Q_eqv_eigenvals[0])**2 + np.imag(Q_eqv_eigenvals[0])**2)
    omega_alpha = np.sqrt(np.real(Q_eqv_eigenvals[2]) ** 2 + np.imag(Q_eqv_eigenvals[2]) ** 2)
    omega_beta = np.sqrt(np.real(Q_eqv_eigenvals[4]) ** 2 + np.imag(Q_eqv_eigenvals[4]) ** 2)
    omega_h_list.append(omega_h)
    omega_alpha_list.append(omega_alpha)
    omega_beta_list.append(omega_beta)
    damping_h_list.append(-np.real(Q_eqv_eigenvals[0])/omega_h)
    damping_alpha_list.append(-np.real(Q_eqv_eigenvals[2]) / omega_alpha)
    damping_beta_list.append(-np.real(Q_eqv_eigenvals[4]) / omega_beta)

    #print(np.real(Q_eqv_eigenvals))




    eig_real_part_list1.append(np.real(Q_eqv_eigenvals[0]))
    eig_real_part_list2.append(np.real(Q_eqv_eigenvals[2]))
    eig_real_part_list3.append(np.real(Q_eqv_eigenvals[4]))



    eig_imag_part_list1.append(np.imag(Q_eqv_eigenvals[0]))
    eig_imag_part_list2.append(np.imag(Q_eqv_eigenvals[1]))
    eig_imag_part_list3.append(np.imag(Q_eqv_eigenvals[2]))
    eig_imag_part_list4.append(np.imag(Q_eqv_eigenvals[3]))
    eig_imag_part_list5.append(np.imag(Q_eqv_eigenvals[4]))
    eig_imag_part_list6.append(np.imag(Q_eqv_eigenvals[5]))


    U_list.append(U)

def tau_hopf(eig_vals, eps = 1e-13):
    """
    :param eig_vals: contains all the eigne values of the system
    :return: tau_hopf (value of the test function
    """

    # Pick out only the complex eigen values
    # eps = 1e-15  # it seems 1e-15 is too low. -> due to numerical errors values larger than 1e-15 can appear
    # eps = 1e-13  # it seems 1e-13 is fine.
    i_cmp = sp.where(sp.absolute(eig_vals.imag) > eps)
    vals_complex = eig_vals[i_cmp]

    # Calculate tau_hopf
    if vals_complex.size == 0:
        tau_h = 0
    else:
        tau_h = 1
        for i, val in enumerate(vals_complex):
            i_cc = sp.where(
                (sp.absolute(val.real - vals_complex.real) < eps) & (sp.absolute(val.imag + vals_complex.imag) < eps))[
                0][0]
            tau_h *= sp.sqrt(val.real + vals_complex[i_cc].real)

    return sp.real(tau_h)

def locate_bifurcation(U_min, U_max):
    """
    Determine the exact location and parameters of the bifurcation point
    :param func: function that returns the eigenvalues of the system
    :param U_min: lower bound of the velocity range [m/s]
    :param U_max: upper bound of the velocity range [m/s]
    :return: bifurcation parameters
    """

    xtol = 1e-5

    tau = lambda U: tau_hopf(ln.eigvals(matrices(1.225, U, 0)))

    # Find the bifurcation point:
    U_bf = opt.brentq(tau, U_min, U_max, xtol=xtol)

    # Calculate bifurcation parameters (frequency, damping ratio):
    Q_eqv = matrices(1.225, U_bf, 0)
    #Q = func(U_bf)
    evals = ln.eigvals(Q_eqv)

    eval_bf = evals[sp.absolute(evals.real).argmin()]

    omega_bf = sp.absolute(eval_bf)
    zeta_bf = -eval_bf.real/sp.absolute(eval_bf)

    return U_bf, omega_bf, zeta_bf, eval_bf

U_bf, omega_bf, zeta_bf, evals = locate_bifurcation(0,200)
print(U_bf, omega_bf, zeta_bf, evals)

# plt.figure()
# plt.xlabel('U in [m/s]')
# plt.ylabel(r'$\omega$')
# plt.plot(U_list, omega_h_list, label='h')
# plt.plot(U_list, omega_alpha_list, label=r'\alpha')
# plt.plot(U_list, omega_beta_list, label=r'\beta')
# plt.grid()
# plt.savefig('omega_plot.png')
# plt.show()
#
# plt.figure()
# plt.xlabel('U in [m/s]')
# plt.ylabel(r'$\zeta$')
# plt.plot(U_list, damping_h_list, label='h')
# plt.plot(U_list, damping_alpha_list, label=r'\alpha')
# plt.plot(U_list, damping_beta_list, label=r'\beta')
# plt.grid()
# plt.savefig('damping_plot.png')
# plt.show()


plt.figure()
plt.scatter(U_list[50:70], eig_imag_part_list1[50:70])
plt.scatter(U_list[50:70], eig_imag_part_list2[50:70])
plt.scatter(U_list[50:70], eig_imag_part_list3[50:70])
plt.scatter(U_list[50:70], eig_imag_part_list4[50:70])
plt.scatter(U_list[50:70], eig_imag_part_list5[50:70])
plt.scatter(U_list[50:70], eig_imag_part_list6[50:70])
plt.title('Imag part eigenvalues')
plt.grid()
plt.show()

plt.figure()
plt.scatter(U_list[50:70], eig_real_part_list1[50:70])
plt.scatter(U_list[50:70], eig_real_part_list2[50:70])
plt.scatter(U_list[50:70], eig_real_part_list3[50:70])
plt.title('Real part eigenvalues')
plt.grid()
plt.show()



