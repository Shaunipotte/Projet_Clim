
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf
from Rayonnement import donnees


#start_date = '2000-06-29 12:00'
#end_date = '2000-06-30 12:00'


def donnees_dynamique(start_date,end_date) : 
    
    filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
    [data, meta] = read_epw(filename, coerce_year=None)
    data
    # Extract the month and year from the DataFrame index with the format 'MM-YYYY'
    month_year = data.index.strftime('%m-%Y')
    # Create a set of unique month-year combinations
    unique_month_years = sorted(set(month_year))
    # Create a DataFrame from the unique month-year combinations
    pd.DataFrame(unique_month_years, columns=['Month-Year'])
    # select columns of interest
    weather_data = data[["temp_air", "dir_n_rad", "dif_h_rad", "relative_humidity"]]
    # replace year with 2000 in the index 
    weather_data.index = weather_data.index.map(
        lambda t: t.replace(year=2000))
    #Pour lire les données à une date et heure précise : 
    weather_data.loc[start_date]
    
    
        # Définition de la durée étudiée 
    
    # Filter the data based on the start and end dates
    weather_data = weather_data.loc[start_date:end_date]
    
    # Remove timezone information from the index
    weather_data.index = weather_data.index.tz_localize(None)
    
    del data
    weather_data
        
    rayonnement = {}
    ouest = {}
    valeur = weather_data.index

    
    dico_dyn = {}
    Text = {}
    phi = weather_data['relative_humidity']
    for val in valeur :
        dico,Tpt = donnees(str(val))
        dico_dyn[str(val)] = dico
        Text[str(val)] = Tpt
    return dico_dyn, Text, phi


def moyenne(start_date,end_date) :
    ray_moyen = {}
    dico, Text = donnees_dynamique(start_date,end_date)
    sommeT = 0
    somme_dir_ouest = 0
    somme_dif_ouest = 0
    somme_ref_ouest = 0
    somme_dir_sud = 0
    somme_dif_sud = 0
    somme_ref_sud = 0
    i = 0
    for key, val in dico.items() :  
        i = i+1
        somme_dir_ouest =  somme_dir_ouest + val['ouest']['dir_rad']
        somme_dif_ouest = somme_dif_ouest + val['ouest']['dif_rad']
        somme_ref_ouest = somme_ref_ouest + val['ouest']['ref_rad']
        somme_dir_sud =  somme_dir_sud + val['sud']['dir_rad']
        somme_dif_sud = somme_dif_sud + val['sud']['dif_rad']
        somme_ref_sud = somme_ref_sud + val['sud']['ref_rad']
        sommeT = sommeT + Text[key]
        
    total_ouest = somme_dir_ouest + somme_dif_ouest + somme_ref_ouest
    ouest = {}
    ouest['dir_rad'] = somme_dir_ouest / i
    ouest['dif_rad'] = somme_dif_ouest / i
    ouest['ref_rad'] = somme_ref_ouest / i
    ouest['total'] = total_ouest/i
    ray_moyen['ouest']=ouest
    
    total_sud = somme_dir_sud + somme_dif_sud + somme_ref_sud
    sud = {}
    sud['dir_rad'] = somme_dir_sud / i
    sud['dif_rad'] = somme_dif_sud / i
    sud['ref_rad'] = somme_ref_sud / i
    sud['total'] = total_sud/i
    ray_moyen['sud']=sud
    
    Tpt_ext = sommeT/i
        
    return ray_moyen, Tpt_ext

#dico_moyen, Text = moyenne(start_date,end_date)
