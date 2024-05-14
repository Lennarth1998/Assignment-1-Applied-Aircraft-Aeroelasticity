import numpy as np
from Constants import * 

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