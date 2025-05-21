# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:09:20 2025

@author: chaou
"""
################ Les imports ###################
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dm4bem import read_epw, sol_rad_tilt_surf, tc2ss, inputs_in_time
from dm4bem import *
from Donnes_dynamiques import donnees_dynamique

###############################################################################
############################# Données #########################################
###############################################################################

start_date = '2000-06-20 12:00:00' # à changer selon la journée que l'on veut
end_date = '2000-06-30 12:00:00'

dico_dyn, Text_dyn = donnees_dynamique(start_date, end_date)
#en dynamique la température change au fur et à mesure

#coeffs d'éclairement
alpha_ext=0.5
alpha_in=0.4
tau=0.3

Amphi = {"largeur": 12,#largeur = direction Nord-Sud
         "longueur":14,#longueur = direction Est-Ouest
         "hauteur":6.5} 

Hall = {"largeur": 3,
       "longueur" : 6,
       "hauteur" : 3} #on ne considère que le RDC, le 1er étage est adiabatique

## définitions de dictionnaires des différents composants
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000,
       'Volume': longueur*largeur*hauteur}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])


concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2}                   # m
     
insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08}                # m

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': 2,
         'Transmission': 0.8}                     # m²

door = {'Conductivity': 0.1,  
        'Width': 0.04,  
       'Surface' : 2}                     # m²

Surface = {'Nord': longueur*hauteur-door['Surface']-glass['Surface'],
           'Sud': longueur*hauteur-glass['Surface'],
           'Milieu':longueur*hauteur-door['Surface'],
           'Adiab':longueur/2*hauteur,
          'Plafond' : longueur*largeur}

### création du panda mur
wall = pd.DataFrame.from_dict({'Layer_in': concrete,
                               'Layer_out': insulation,
                               'Glass': glass,
                                'Door': door},
                              orient='index')

# définition coeff convection 
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])

#thermostat ######
KpN = 1e-4 #pièce nord, no controller Kp -> 0
KpS = 1e3 #pièce Sud, almost perfect controller Kp -> ∞
deltaT_h = 15 # différence de température en hiver
deltaT_s = -10 # différence de température en été
Tc = 18 #été
#Tc = 21 #hiver

## flux utilisateur
Qa = 80 #~80 par personne, ici c'est celui de la pièce Nord (four, télé, personnes)

###############################################################################
############################# Le schéma général ###############################
###############################################################################
#les noeuds
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14']
# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20']
# temperature nodes
nθ = len(θ)      # number of temperature nodes
# flow-rate branches
nq = len(q)     # number of flow branches


########################### matrice A des flux #############################

A = np.zeros([nq, nθ])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5

A[6, 5], A[6, 6] = 1, -1    # branch 6: node 5 -> node 6
A[7, 6], A[7, 7] = 1, -1    # branch 7: node 6 -> node 7
A[8, 7], A[8, 8] = 1, -1    # branch 8: node 7 -> node 8
A[9, 8], A[9, 9] = 1, -1    # branch 9: node 8 -> node 9
A[10, 9], A[10, 10] = 1, -1    # branch 10: node 9 -> node 10
A[11, 10], A[11, 11] = 1, -1    # branch 11: node 10 -> node 11
A[12, 11], A[12, 12] = 1, -1    # branch 12: node 11 -> node 12
A[13, 12], A[13, 13] = 1, -1    # branch 13: node 12 -> node 13
A[14, 13], A[14, 14] = 1, -1    # branch 14: node 13 -> node 14
A[15, 14]= 1   # branch 15: node 14 -> node 15

# porte, fenetre, ventilation
A[18, 5]= 1
A[17, 5]= 1
A[17, 9]= -1
A[16, 9]= 1

#controler
A[19,5] = 1
A[20,9] = 1

#ponts thermiques


A = pd.DataFrame(A, index=q, columns=θ)

############ Matrice B avec T_ext définit à l'aide du code rayonnement ########
b = pd.Series(['Text', 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 'Text', 'Text', 0,
               'Text', 'Tc', 'Tc'],
              index=q)

#################################### Matrice G ################################

# définiton conductance de conduction
G_cd = wall['Conductivity'] / wall['Width']
pd.DataFrame(G_cd, columns=['Conductance'])
##G de des infiltrations d'air pour les différentes parois
ACH = {'S': 2, 
       'I': 2,
      'N':4}
Va_dot = {'S' : ACH['S'] / 3600 * air['Volume'],
              'I' : ACH['I'] / 3600 * air['Volume'],
              'N' : ACH['N'] / 3600 * air['Volume']}
Gv = {'S' :  air['Density'] * air['Specific heat'] * Va_dot['S'],
       'I' : air['Density'] * air['Specific heat'] * Va_dot['I'],
       'N' : air['Density'] * air['Specific heat'] * Va_dot['N']} 

#Gv['S'] = 0 ##ventilation nord vers Sud
Gv['N'] = 0  ##ventilation Sud vers Nord

# glass: convection outdoor & conduction
Gglass16 = wall.loc['Glass', 'Surface'] / (1 / h['out'] + 1 / G_cd['Glass'] + 1 / h['in'])
Gporte16 = wall.loc['Door', 'Surface'] / (1 / h['out'] + 1 / G_cd['Door'] + 1 / h['in'])
Gporte17 = wall.loc['Door', 'Surface'] / (1 / h['in'] + 1 / G_cd['Door'] + 1 / h['in'])
G16 = float(Gv['S'] + Gglass16.iloc[0])
G17 = float(Gv['I'] + Gporte17.iloc[0])
G18 = float(Gv['N'] + Gglass16.iloc[0] + Gporte16.iloc[0])

## remplissage de G
GN = np.array(np.hstack([h['out'].iloc[0] * Surface['Nord'], 
      G_cd['Layer_out']*Surface['Nord']/2,
      G_cd['Layer_out']*Surface['Nord']/2,
      G_cd['Layer_in']*Surface['Nord']/2,
      G_cd['Layer_in']*Surface['Nord']/2,
      h['in'].iloc[0] * Surface['Nord']]))

GM = np.array((h['in'].iloc[0] * Surface['Milieu'],
      G_cd['Layer_in']*Surface['Milieu']/2,
      G_cd['Layer_in']*Surface['Milieu']/2,
      h['in'].iloc[0] * Surface['Milieu']))

GS = np.array((h['in'].iloc[0] * Surface['Sud'],
      G_cd['Layer_in']*Surface['Sud']/2,
      G_cd['Layer_in']*Surface['Sud']/2,
      G_cd['Layer_out']*Surface['Sud']/2,
      G_cd['Layer_out']*Surface['Sud']/2,
      h['out'].iloc[0] * Surface['Sud']))

GP = np.array((G16, G17, G18))
GC = np.array((KpN, KpS))

G = np.array(np.hstack((GN, GM, GS, GP, GC)))
G = pd.DataFrame(G, index=q)

########################## Matrice f des flux apportés ########################
f = pd.Series(['Φin', 0, 0, 0, 'ΦiN1', 'Qa', 
               'ΦiN2', 0, 'ΦiO2', 0, 'ΦiO1',
               0, 0, 0, 'ΦiO'],
              index=θ)

############# Matrice C des capacités (en statique non utile) #################

# Compute capacities for walls
C_walls = wall['Density'] * wall['Specific heat'] * wall['Width']
# Compute capacity for air
C_air = air['Density'] * air['Specific heat'] * air['Volume']

# Assign non-zero capacities to specific diagonal elements
CN = np.array(np.hstack([0,
                         C_walls.loc['Layer_out']*Surface['Nord'],
                         0,
                         C_walls.loc['Layer_in']*Surface['Nord'],
                         0,
                         C_air]))
CM = np.array(np.hstack([0,
                         C_walls.loc['Layer_in']*Surface['Milieu'], 
                         0,
                         C_air]))
CS = np.array(np.hstack([0,
                         C_walls.loc['Layer_in']*Surface['Sud'],
                         0,
                         C_walls.loc['Layer_out']*Surface['Sud'],
                         0]))


C = np.array(np.hstack((CN, CM, CS)))
C = pd.DataFrame(C, index=θ)

# Matrice des températures
y = np.zeros([len(θ)])     # nodes and len(θ) = 15
pd.DataFrame(y, index=θ)


###############################################################################
###################### Résolution dynamique ###################################
###############################################################################

# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.DataFrame(G, index=q)
C = pd.DataFrame(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)
y.loc[["θ1", "θ3", "θ5", "θ7", "θ9", "θ11", "θ13"]] = 1

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

#on se retrouve avec ce circuit : thermal circuit
print("A:", A.shape)
print("G:", G.shape)
print("C:", C.shape)
print("b:", b.shape)
print("f:", f.shape)
print("y:", y.shape)

## système  DAE : utliser dm4bem
[As, Bs, Cs, Ds, us] = tc2ss(TC)

################################### discretisation #######################################

#définition du pas de temps
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)        #pas de temps max
print(f"Pas de temps maximal pour stabilité : {dtmax:.2f} s") 

dt = 160 # Choisir un pas de temps inférieur pour garantir la stabilité

# settling time, temps de fin de simulation max
t_f = 4 * max(-1 / λ)
print_rounded_time('t_settle', t_f) 

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_f / 3600) * 3600
print_rounded_time('duration', duration)


#maintenant on passe à la définition des points et du dico de u (f et T)
n_points = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).total_seconds() / dt)
n = n_points

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start = start_date,
                           periods = n, freq=f"{int(dt)}S")

#les températures
Text = np.ones(n)
k = int(3600/dt) #nombre de T_dyn qui se répètent car correspondent à 1 h
v=0
for key, value in Text_dyn.items():
    Text[v : v+k] = value
    v = v+k

Tch = Text
Tc = Tc*np.ones(n)

Qa = 90*np.ones(n) # on peut tenter ensuite de simuler une évolution des consommations selon la nuit ou le jour en remplissant avec une boucle
nuit = int((3600*9/dt))
jour = int((3600*18/dt))
Qa[0:0+nuit] = 120
Qa[nuit:jour] = 70

#les flux au nord
Φin = np.ones(n)
ΦiN1 = np.ones(n)
ΦiN2 = np.ones(n)
v=0
for key, value in dico_dyn.items():
    EN = value['nord']['total']
    Φin[v : v+k] = alpha_ext*EN*Surface["Nord"]
    ΦiN1[v : v+k] = alpha_in*tau*EN*glass["Surface"]*(Surface["Nord"]/(Surface["Milieu"]+2*Surface["Adiab"]+Surface["Nord"]+2*Surface["Plafond"]))
    ΦiN2[v : v+k] = alpha_in*tau*EN*glass["Surface"]*(Surface["Milieu"]/(Surface["Milieu"]+2*Surface["Adiab"]+Surface["Nord"]+2*Surface["Plafond"]))
    v = v+k

#ceux ç l'ouest
Φio = np.ones(n)
Φio1 = np.ones(n)
Φio2 = np.ones(n)
v=0
for key, value in dico_dyn.items():
    ES = value['ouest']['total']
    Φio[v : v+k] = alpha_ext*ES*Surface["Sud"]
    Φio1[v : v+k] = alpha_in*tau*ES*glass["Surface"]*(Surface["Sud"]/(2*Surface["Adiab"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))
    Φio2[v : v+k] = alpha_in*tau*ES*glass["Surface"]*(Surface["Milieu"]/(2*Surface["Adiab"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))
    v = v+k

data = {'Text': Text, 'Tc': Tc, 'Φin': Φin, 'ΦiN1': ΦiN1, 'Qa': Qa, 'ΦiN2': ΦiN2, 'ΦiO2': Φio2, 'ΦiO1': Φio1, 'ΦiO': Φio}
input_data_set = pd.DataFrame(data, index=time)

u = inputs_in_time(us, input_data_set)


################################## ENFIN CALCULER LES CHOSES QUON VEUT #####################
# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 20   # initial temperatures 

θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])

# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1,)
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

#flux des controlleurs seulement
u['q19'] = pd.to_numeric(u['q19'], errors='coerce')
y['θ5'].iloc[:, 0] = pd.to_numeric(y['θ5'].iloc[:, 0], errors='coerce')
y['θ5'].iloc[:, 1] = pd.to_numeric(y['θ5'].iloc[:, 1], errors='coerce')
y['θ9'].iloc[:, 0] = pd.to_numeric(y['θ9'].iloc[:, 0], errors='coerce')
y['θ9'].iloc[:, 1] = pd.to_numeric(y['θ9'].iloc[:, 1], errors='coerce')

#pièce Nord
Sn = 2*Surface["Plafond"]+Surface["Nord"]+Surface["Milieu"] # m², surface area of the house
q_HVAC_N_exp = KpN * (u['q19'] - y['θ5'].iloc[:, 0]) / Sn  # W/m²
q_HVAC_N_imp = KpN * (u['q19'] - y['θ5'].iloc[:, 1]) / Sn  # W/m²
#pièce Nord
Ss = 2*Surface["Plafond"]+Surface["Sud"]+Surface["Milieu"]  # m², surface area of the house
q_HVAC_S_exp = KpS * (u['q20'] - y['θ9'].iloc[:, 0]) / Ss  # W/m²
q_HVAC_S_imp = KpS * (u['q20'] - y['θ9'].iloc[:, 1]) / Ss  # W/m²

#on met ça dans un tableau
Q = pd.DataFrame(index=u.index)
Q['q_HVAC_N_exp'] = q_HVAC_N_exp
Q['q_HVAC_S_exp'] = q_HVAC_S_exp
Q['q_HVAC_N_imp'] = q_HVAC_N_imp
Q['q_HVAC_S_imp'] = q_HVAC_S_imp

##################################### Plot des choses ################################
# Créer les styles pour chaque série
linestyles = ['-'] * 7 + ['--'] * 7  # Traits solides pour les 7 premiers, pointillés pour les 7 derniers

# Définir les noms des courbes
labels = ['$\\theta_1$', '$\\theta_3$', '$\\theta_5$', '$\\theta_7$', '$\\theta_9$', 
    '$\\theta_{11}$', '$\\theta_{13}$', '$\\theta_1$ imp', '$\\theta_3$ imp', 
    '$\\theta_5$ imp', '$\\theta_7$ imp', '$\\theta_9$ imp', 
    '$\\theta_{11}$ imp', '$\\theta_{13}$ imp']
colors = ['#FF0000','#FFD700','#00FF00','#0000FF','#FF4500', '#800080', '#FF1493','#800000',
          '#808000','#008000','#000080','#8B0000','#0000A0','#800080']

####################################### La figure ######################################
# Create figure with increased size
fig, ax = plt.subplots(4, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing

# on sépare les deux analyses
for i in range(len(y.columns)):
    if i not in [2, 4, 9, 11]:
        y_col = y.iloc[:, i]
        ax[0].plot(y_col.index, y_col, label=labels[i], linestyle=linestyles[i], color=colors[i])
    else:
        y_col = y.iloc[:, i]
        ax[1].plot(y_col.index, y_col, label=labels[i], linestyle=linestyles[i], color=colors[i])

# les flux HVAC
Q[['q_HVAC_N_exp', 'q_HVAC_S_exp', 'q_HVAC_N_imp', 'q_HVAC_S_imp']].plot(ax=ax[2])
#les temps ext
text_series = pd.Series(Text_dyn).sort_index()
ax[3].plot(text_series.index, text_series.values)

# Configure subplot 1 (les parois)
ax[0].set_xlabel('Time', fontsize=12)
ax[0].set_ylabel('Temperature $\\theta_i$ (°C)', fontsize=12)
ax[0].set_title(f'CAS 0 - Wall Temperatures: $dt$ = {dt:.0f} s, $dt_{{max}}$ = {dtmax:.0f} s, CI: {θ0}', fontsize=14)
ax[0].legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10)
ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=24))#permet de pas avoir 100000 valeurs de temps sur le graphique, à changer si pas 10jours
ax[0].grid(True, linestyle='--', alpha=0.7)

# Configure subplot 2 (les pièces)
ax[1].set_xlabel('Time', fontsize=12)
ax[1].set_ylabel('Temperature $\\theta_i$ (°C)', fontsize=12)
ax[1].set_title(f'CAS 0 - Room Temperatures: $dt$ = {dt:.0f} s, CI: {θ0}', fontsize=14)
ax[1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10)
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=24))#permet de pas avoir 100000 valeurs de temps sur le graphique, à changer si pas 10jours
ax[1].grid(True, linestyle='--', alpha=0.7)

# Configure subplot 3 (les flux de radiateurs)
ax[2].set_ylabel('Heat Rate $q$ (W·m⁻²)', fontsize=12)
ax[2].set_xlabel('Time', fontsize=12)
ax[2].set_title(f'CAS 0 - HVAC Fluxes: $dt$ = {dt:.0f} s', fontsize=14)
ax[2].legend(['$q_{HVAC} North$ Exp.', '$q_{HVAC} South$ Exp.', 
              '$q_{HVAC} North$ Imp.', '$q_{HVAC} South$ Imp.'], 
             bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10)
ax[2].grid(True, linestyle='--', alpha=0.7)

# Configure subplot 4 (les températures extérieures)
ax[3].set_xlabel('Time', fontsize=12)
ax[3].set_ylabel('Temperature (°C)', fontsize=12)
ax[3].set_title('CAS 0 - Outside Temperatures', fontsize=14)
ax[3].legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10)
ax[3].xaxis.set_major_locator(mdates.HourLocator(interval=24*60)) #permet de pas avoir 100000 valeurs de temps sur le graphique, à changer si pas 10jours
ax[3].grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.show()
