import numpy as np

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