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
from psychro import w

###############################################################################
############################# Données #########################################
###############################################################################

#choisir la saison

#saison = "hiver"
saison = "été"

# à changer selon la saison que l'on veut
start_date = '2000-06-20 08:00:00' 
end_date = '2000-06-24 08:00:00'



###################### conditions inchangées #################
dico_dyn, Text_dyn, phi_dyn = donnees_dynamique(start_date, end_date)
#en dynamique la température change au fur et à mesure
phie_h = 0.4                        # humidité relative en hiver
phie_e = 0.7                        # été
phii = 0.5                          # controle humidité intérieure
phi = {"Ext_hiver" : 0.4,
       "Ext_ete" : 0.7,
       "Int" : 0.5}                             

#coeffs d'éclairement
alpha_ext=0.5
alpha_in=0.4
tau=0.3

#########   considérations géométriques ###########################################
Amphi = {"largeur": 12,                     #largeur = direction Nord-Sud
         "longueur":14,                     #longueur = direction Est-Ouest
         "hauteur":6.5} 

Hall = {"largeur": 3,
       "longueur" : 8,
       "hauteur" : 3}                       #on ne considère que le RDC, le 1er étage est adiabatique

## définitions de dictionnaires des différents composants
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000,               # J/(kg·K)
       'Volume_hall': Hall["largeur"]*Hall["longueur"]*Hall["hauteur"],
       'Volume_amphi' : Amphi["largeur"]*Amphi["longueur"]*Amphi["hauteur"]}               
pd.DataFrame(air, index=['Air'])


concrete = {'Conductivity': 1.75,           # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.175}                 # m
     
insulation = {'Conductivity': 0.004,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.04}                # m

glass = {'Conductivity': 1.4,                         # W/(m·K)
         'Density': 2500,                             # kg/m³
         'Specific heat': 1210,                       # J/(kg⋅K)
         'Width': 0.04,                               # m
         'Surface': Hall["longueur"]*Hall["hauteur"], #l'accès extérieur est que du verre
         'Transmission': 0.8}                         # m²

door = {'Conductivity': 0.1,  
        'Width': 0.04,  
       'Surface' : 2*3}                            # m²
door_secours = {'Conductivity': 45,  
        'Width': 0.04,  
       'Surface' : 2*1.5}                            # m²

Surface = {'A_ouest': Amphi["largeur"]*Amphi["hauteur"],
           'A_adiab': 2*Amphi["longueur"]*Amphi["hauteur"],
           'A_Plafond' : Amphi["longueur"]*Amphi["largeur"],
           'H_adiab':Hall["longueur"]*Hall["hauteur"]+Hall["largeur"]*Hall["hauteur"],
           'H_sud':glass["Surface"],
           'H_Plafond' : Hall["longueur"]*Hall["largeur"],
          'Interface' : Hall["largeur"]*Hall["hauteur"]-door["Surface"]}

### création du panda mur
wall = pd.DataFrame.from_dict({'Layer_in': concrete,
                               'Layer_out': insulation,
                               'Glass': glass,
                               'Door': door,
                               'Issue_secours' : door_secours},
                              orient='index')

################ convection ################################### 
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])

###################################### CTA ###############################
KpH = 1e-4                      #HALL, no controller Kp -> 0
KpA = 1e4                       #AMPHI, almost perfect controller Kp -> ∞
deltaT_h = 15                   # différence de température en hiver
deltaT_e = -10                  # différence de température en été

#################### charges auxiliaires ##################################
npe = 70               # personnes en été
nph = 40               # personnes en hiver
npmin = 3              # dans le hall
#sensible
Qsa_h = 92*nph          # personnes en amphi en hiver, on sous-évalue
Qsa_e = 77*npe          # personnes amphi en été, on étudie quel mois ?
Qshall = 80*npmin       # On considère qu'il y a toujours 3 personnes dans le hall
Qled = 1400             # éclairage
#latent
Qla_h = 27*nph          # personnes en amphi en hiver, on sous-évalue
Qla_e = 41*npe          # personnes amphi en été, on étudie quel mois ?
Qlhall = 30*npmin       

########## Infiltrations permanentes #############################
Va_tot = 300                # m3/h
Va_ha = 280                 # hall-amphi
ACH = {'Amphi_o': (Va_tot-Va_ha)/air['Volume_amphi'], 
       'Interface':(Va_ha)/air['Volume_amphi'],
      'Hall_s': 6}             #on considère une ventilation importante avec la porte automatique
Va_dot = {'Amphi' : (Va_tot-Va_ha)/3600,
          'Interface' : Va_ha/3600,
          'Hall' : ACH['Hall_s'] / 3600 * air['Volume_hall']}


############### Ponts thermiques ################################
psy_s = 1.36

###############################################################################
############################# Le schéma général ###############################
###############################################################################
#les noeuds
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 
     'θ12', 'θ13', 'θ14']                                                          # temprature nodes 
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
     'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22']  # flow-rate branches
nθ = len(θ)                                                                        # number of temperature nodes
nq = len(q)                                                                        # number of flow branches


########################### matrice A des flux #############################

A = np.zeros([nq, nθ])          # n° of branches X n° of nodes

#amphi
A[0, 0] = 1                     # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1        # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1        # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1        # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1        # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1        # branch 5: node 4 -> node 5

#interface
A[6, 5], A[6, 6] = 1, -1        # branch 6: node 5 -> node 6
A[7, 6], A[7, 7] = 1, -1        # branch 7: node 6 -> node 7
A[8, 7], A[8, 8] = 1, -1        # branch 8: node 7 -> node 8
A[9, 8], A[9, 9] = 1, -1        # branch 9: node 8 -> node 9

#hall
A[10, 9], A[10, 10] = 1, -1     # branch 10: node 9 -> node 10
A[11, 10], A[11, 11] = 1, -1    # branch 11: node 10 -> node 11
A[12, 11], A[12, 12] = 1, -1    # branch 12: node 11 -> node 12
A[13, 12], A[13, 13] = 1, -1    # branch 13: node 12 -> node 13
A[14, 13], A[14, 14] = 1, -1    # branch 14: node 13 -> node 14
A[15, 14]= 1                    # branch 15: node 14 -> node 15

# porte, fenetre, ventilation
A[18, 5]= 1
A[17, 5]= 1
A[17, 9]= -1
A[16, 9]= 1

#controler
A[19,5] = 1
A[20,9] = 1

#ponts thermiques
A[21,5] = 1
A[22,9] = 1

A = pd.DataFrame(A, index=q, columns=θ)

############### Matrice B  #########################
b = pd.Series(['Text', 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 'Text', 'Text', 0,
               'Text', 'Ts', 'Ts', 'Text', 'Text'],
              index=q)
#avec T_ext défini à l'aide du code rayonnement
#Ts, température de soufflage de la CTA

#################################### Matrice G ################################

# définiton conductance
G_cd = wall['Conductivity'] / wall['Width']
pd.DataFrame(G_cd, columns=['Conductance'])

#G des infiltrations d'air pour les différentes parois
Gv = {'A' :  air['Density'] * air['Specific heat'] * Va_dot['Amphi'],
       'I' : air['Density'] * air['Specific heat'] * Va_dot['Interface'],
       'H' : air['Density'] * air['Specific heat'] * Va_dot['Hall']} 
#Gv['H'] = 0            ##ventilation Amphi vers Hall
Gv['A'] = 0             ##ventilation Hall vers Amphi

# Les résistances en parallèle
Gglass16 = wall.loc['Glass', 'Surface'] / (1 / h['out'] + 1 / G_cd['Glass'] + 1 / h['in']) #on le garde au-cas où on veut faire un autre cas
Gporte18 = wall.loc['Issue_secours', 'Surface'] / (1 / h['out'] + 1 / G_cd['Issue_secours'] + 1 / h['in'])
Gporte17 = wall.loc['Door', 'Surface'] / (1 / h['in'] + 1 / G_cd['Door'] + 1 / h['in'])
G16 = float(Gv['H'])
G17 = float(Gv['I'] + Gporte17.iloc[0])
G18 = float(Gv['A'] + Gporte18.iloc[0])

GP = np.array((G16, G17, G18))

## remplissage de G, les murs
GA_o = np.array(np.hstack([h['out'].iloc[0] * Surface['A_ouest'], 
      G_cd['Layer_out']*Surface['A_ouest']/2,
      G_cd['Layer_out']*Surface['A_ouest']/2,
      G_cd['Layer_in']*Surface['A_ouest']/2,
      G_cd['Layer_in']*Surface['A_ouest']/2,
      h['in'].iloc[0] * Surface['A_ouest']]))

GI = np.array((h['in'].iloc[0] * Surface['Interface'],
      G_cd['Layer_in']*Surface['Interface']/2,
      G_cd['Layer_in']*Surface['Interface']/2,
      h['in'].iloc[0] * Surface['Interface']))

GH_s = np.array((h['in'].iloc[0] * Surface['H_sud'],
      G_cd['Glass']*Surface['H_sud']/2,               #j'ai gardé ça pour pouvoir utiliser les capacités après
      G_cd['Glass']*Surface['H_sud']/2,
      0,
      0,
      h['out'].iloc[0] * Surface['H_sud']))

## controller (CTA)
GC = np.array((KpA, KpH))

# ponts thermiques
GTH_amphi = psy_s*Amphi["largeur"]
GTH_hall = psy_s*Hall["longueur"]
GTH = np.array((GTH_amphi, GTH_hall))

G = np.array(np.hstack((GA_o, GI, GH_s, GP, GC, GTH)))
G = pd.DataFrame(G, index=q)

########################## Matrice f des flux apportés ########################
if saison == "hiver":
    QsA = Qsa_h+Qled
    QlA = Qla_h
else :
    QsA = Qsa_e+Qled
    QlA = Qla_e
    
f = pd.Series(['ΦiO', 0, 0, 0, 'ΦiO1', 'QaA', 
               'ΦiO2', 0, 'ΦiS2', 'QaH', 'ΦiS1',
               0, 0, 0, 'ΦiS'],
              index=θ)

############# Matrice C des capacités (en statique non utile) #################

# Compute capacities for walls
C_walls = wall['Density'] * wall['Specific heat'] * wall['Width']
# Compute capacity for air
C_air_a = air['Density'] * air['Specific heat'] * air['Volume_amphi']
C_air_h = air['Density'] * air['Specific heat'] * air['Volume_hall']
# Compute capacity for glass
C_glass = glass['Density'] * glass['Specific heat'] * glass['Width']

# Assign non-zero capacities to specific diagonal elements
CA_o = np.array(np.hstack([0,
                         C_walls.loc['Layer_out']*Surface['A_ouest'],
                         0,
                         C_walls.loc['Layer_in']*Surface['A_ouest'],
                         0,
                         C_air_a]))
CI = np.array(np.hstack([0,
                         C_walls.loc['Layer_in']*Surface['Interface'], 
                         0]))
CH_s = np.array(np.hstack([C_air_h, 
                           0,
                         C_walls.loc['Layer_in']*Surface['H_sud'],
                         0,
                         C_walls.loc['Layer_out']*Surface['H_sud'],
                         0]))


C = np.array(np.hstack((CA_o, CI, CH_s)))
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

########### MATRICE b : les températures et humidités sur la durée voulue ########################

Text = np.ones(n)
k = int(3600/dt) #nombre de T_dyn qui se répètent car correspondent à 1 h
v=0
for key, value in Text_dyn.items():
    Text[v : v+k] = value
    v = v+k

#controler
if saison == "hiver":
    dTc = deltaT_h
else : 
    dTc = deltaT_e
Tc = Text+dTc

#relative humidity
phi_e = np.ones(n)
k = int(3600/dt) #nombre de T_dyn qui se répètent car correspondent à 1 h
v=0
for key, value in phi_dyn.items():
    phi_e[v : v+k] = value
    v = v+k


######################## MATRICE f : les flux ##################################################
QaA = np.ones(n)
QaH = np.ones(n)

#pas de prise en compte fermeture départ
Qa = 90*np.ones(n) # on peut tenter ensuite de simuler une évolution des consommations selon la nuit ou le jour en remplissant avec une boucle

#fermeture départ
repos = int((3600*14/dt))
cours = int((3600*10/dt))
QaA[0:0+cours] = QsA
QaA[cours:repos] = 0

QaH[0:0+cours] = Qshall
QaH[cours:repos] = 0

#les flux amphi à l'ouest
ΦiO = np.ones(n)
ΦiO1 = np.ones(n)
ΦiO2 = np.ones(n)
v=0
for key, value in dico_dyn.items():
    EO = value['ouest']['total']
    ΦiO[v : v+k] = alpha_ext*EO*Surface["A_ouest"]
    ΦiO1[v : v+k] = alpha_in*tau*EO*0*(Surface["A_ouest"]/(Surface["Interface"]+2*Surface["A_adiab"]+Surface["A_ouest"]+2*Surface["A_Plafond"]))
    ΦiO2[v : v+k] = alpha_in*tau*EO*0*(Surface["Interface"]/(Surface["Interface"]+2*Surface["A_adiab"]+Surface["A_ouest"]+2*Surface["A_Plafond"]))
    v = v+k

#ceux au hall au sud
ΦiS = np.ones(n)
ΦiS1 = np.ones(n)
ΦiS2 = np.ones(n)
v=0
for key, value in dico_dyn.items():
    ES = value['sud']['total']
    ΦiS[v : v+k] = alpha_ext*ES*Surface["H_sud"]
    ΦiS1[v : v+k] = alpha_in*tau*ES*glass["Surface"]*(Surface["H_sud"]/(2*Surface["H_adiab"]+Surface["Interface"]+Surface["H_sud"]+2*Surface["H_Plafond"]))
    ΦiS2[v : v+k] = alpha_in*tau*ES*glass["Surface"]*((Hall["longueur"]*Hall["hauteur"])/(2*Surface["H_adiab"]+Surface["Interface"]+Surface["H_sud"]+2*Surface["H_Plafond"]))
    v = v+k

data = {'Text': Text, 'Ts': Tc, 'ΦiO': ΦiO, 'ΦiO1': ΦiO1, 'QaA': QaA, 'ΦiO2': ΦiO2, 'ΦiS2': ΦiS2, 'ΦiS1': ΦiS1, 'ΦiS': ΦiS, 'QaH' : QaH}
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


############################### C T A ########################################"
#flux des controlleurs seulement
u['q19'] = pd.to_numeric(u['q19'], errors='coerce')
y['θ5'].iloc[:, 0] = pd.to_numeric(y['θ5'].iloc[:, 0], errors='coerce')
y['θ5'].iloc[:, 1] = pd.to_numeric(y['θ5'].iloc[:, 1], errors='coerce')
y['θ9'].iloc[:, 0] = pd.to_numeric(y['θ9'].iloc[:, 0], errors='coerce')
y['θ9'].iloc[:, 1] = pd.to_numeric(y['θ9'].iloc[:, 1], errors='coerce')

#Amphi
SA = 2*Surface["A_Plafond"]+Surface["A_ouest"]+Surface["A_adiab"]+Surface["Interface"] # m², surface area of the house
q_HVAC_O_exp = KpA * (u['q19'] - y['θ5'].iloc[:, 0]) / SA  # W/m²
q_HVAC_O_imp = KpA * (u['q19'] - y['θ5'].iloc[:, 1]) / SA  # W/m²
#Hall
SH = 2*Surface["H_Plafond"]+Surface["H_sud"]+Surface["Interface"]+Surface["H_adiab"]  # m², surface area of the house
q_HVAC_S_exp = KpH * (u['q20'] - y['θ9'].iloc[:, 0]) / SH  # W/m²
q_HVAC_S_imp = KpH * (u['q20'] - y['θ9'].iloc[:, 1]) / SH  # W/m²

#on met ça dans un tableau

#charges sensibles
Qs = pd.DataFrame(index=u.index)
Qs['q_HVAC_O_exp'] = q_HVAC_O_exp
Qs['q_HVAC_S_exp'] = q_HVAC_S_exp
Qs['q_HVAC_O_imp'] = q_HVAC_O_imp
Qs['q_HVAC_S_imp'] = q_HVAC_S_imp

#renouvellement CTA 
Mdot = pd.DataFrame(index=u.index)


Ql = pd.DataFrame(index = u.index)

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
Q[['q_HVAC_O_exp', 'q_HVAC_S_exp', 'q_HVAC_O_imp', 'q_HVAC_S_imp']].plot(ax=ax[2])
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
