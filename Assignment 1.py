import numpy as np
import matplotlib.pyplot as plt

U=1 #NEEDS CHANGING!!!
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
T3              = -(1/8+ch**2)*mu**2 + 1/4*ch*np.sqrt(1-ch**2)*mu*(7+ch**2)-1/8 * (1-ch**2)*(5 *ch**2+4)

T4              = -mu + ch*np.sqrt(1-ch**2)
T5              = -(1-ch**2)-mu**2+ 2*ch*np.sqrt(1-ch**2)*mu
T6              = T2

T7              = -(1/8+ch**2)*mu + 1/8*ch*np.sqrt(1-ch**2)*(7+ch**2)
T8              = -1/3 * np.sqrt(1-ch**2)*(2*ch**2 + 1)+ch*mu
T9              = 1/2 * (1/3*(np.sqrt(1-ch**2))**3+a*T4)

T10             = np.sqrt(1-ch**2)+mu
T11             = mu*(1-2*ch) + np.sqrt(1-ch**2)*(2-ch)
T12             = np.sqrt(1-ch**2)*(2+ch)-mu*(2*ch + 1)

T13             = 1/2*(-T7 - (ch-a)*T1)
T14             = 1/16 + 1/2 * a*ch


# Structual mass matrix [A}:
A = np.array([[m , S , S_beta],
             [S , I_beta, I_alpha_beta],
             [S_beta,I_alpha_beta, I_beta]])

# Aerodynamic mass matrix [B]:
B = b**2 * np.array([[np.pi, -np.pi *a*b, -T1*b],
                     [-np.pi*a*b, np.pi*b**2*(1/8 + a**2), -(T7+(ch-a)*T1)*b**2],
                     [-T1*b, 2*T13*b**2, (-T3*b**2)/np.pi]])

#Mass matrix [M]:
M = A + rho*B 
    
#Structual Damping [C]:
C = np.array([[C_h, 0 , 0],
    [0 , C_alpha, 0],
    [0 , 0 , C_beta]])

# Aerodynamic Damping [D]:
D1 = b**2 * np.array([[0, np.pi, -T4],
                     [0, np.pi*(1/2 -a)*b, (T1-T8-(ch-a)*T4 +T11/2)*b],
                     [0, (-2*T9 -T1 + T4*(a-1/2))*b, (-T4*T11*b)/(2*np.pi)]])

D2 = np.array([[2*np.pi*b,  2*np.pi*b**2*(1/2-a), b**2*T11],
               [-2*np.pi*b**2*(a+1/2), -2*np.pi*b**3*(a+1/2)*(1/2-a), -b**3*(a+1/2)*T11],
               [b**2*T12, b**3*T12*(1/2-a), (b**3*T12*T11)/(2*np.pi)]])

D = D1 + (1-psi_1-psi_2)*D2


#Structual stiffness matrix [E]:
E = np.array([[K_h, 0 , 0],
    [0 , K_alpha, 0],
    [0 , 0 , K_beta]])

#Aerodynamic stiffness matrix [F]:
F1 = b**2 * np.array([[0,0,0],
                      [0,0,(T4+T10)],
                      [0,0,(T5-T4*T10)/np.pi]])

F2 = np.array([[0,2*np.pi*b, 2*b*T10],
               [0,-2*np.pi*b**2*(a+1/2),-2*b**2*(a+1/2)*T10],
               [0,b**2*T12, (b**2*T12*T10)/np.pi]])

F3 = np.array([[2*np.pi*b, 2*np.pi*b**2*(1/2-a),b**2*T11],
               [-2*np.pi*b**2*(a+ 1/2), -2*np.pi*b**3*(a+ 1/2)*(1/2-a), -b**3*(a+1/2)*T11],
               [b**2*T12, b**3*T12*(1/2-a), (b**3*T12*T11)/2*np.pi]])

F= F1+(1-psi_1-psi_2)*F2+ ((psi_1*epsi_1)/b+(psi_2*epsi_2)/b)*F3

#Aerodynamic influence matrix [W]:
W0 = np.array([[-psi_1*(-epsi_1/b)**2],
               [-psi_2*(-epsi_2/b)**2],
               [psi_1*epsi_1*(1-epsi_1*(1/2-a))/b],
               [psi_2*epsi_2*(1-epsi_2*(1/2-a))/b],
               [psi_1*epsi_1*(T10-(epsi_1*T11)/2)/(np.pi*b)],
               [psi_2*epsi_2*(T10-(epsi_2*T11)/2)/(np.pi*b)]])


W = np.array([[2*np.pi*W0.T],
             [-2*np.pi*b**2*(a+1/2)*W0.T],
             [b**2*T12*W0.T]])
W = W.reshape([3,6])

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

Q = np.block([[M_inv @ (C+rho*U*D),-M_inv @ (E+rho*U**2*F),-rho*U**3*M_inv @ W],
              [np.eye(3),np.zeros([3,3]),np.zeros([3,6])],
              [np.zeros([6,3]), W1,U*W2]])

q_n = np.block([[-M_inv @ [[1],
                         [0],
                         [0]]],
                 [np.zeros([9,1])]])



