# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 13:27:18 2025

@author: pc
"""

from pyomeca import Markers as py
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
import xarray as xr
import csv
import sys
import re

#%% Fonctions
def Lire_fichier (fichier) : 
    df = pd.read_csv(fichier, sep=';')
    return df

def get_marker(rep, channel):
    try:
        return np.array(rep.sel(channel=channel, axis=['x', 'y', 'z']))
    except KeyError:
        print(f"Marqueur '{channel}' manquant.")
        return None
    
def angle_degre(v1,v2): # Obj déterminer l'angle en degré entre 2 segments corporelle
    ps = (v1[0]*v2[0])+(v1[1]*v2[1])+(v1[2]*v2[2]) # produit scalaire des 2 vecteur > v1@v2 mais manuellement pour toute les lignes
    n1 = np.sqrt((v1[0]*v1[0])+(v1[1]*v1[1])+(v1[2]*v1[2])) # normes > np.sqrt(v1@v1) mais manuellement
    n2 = np.sqrt((v2[0]*v2[0])+(v2[1]*v2[1])+(v2[2]*v2[2]))
    return np.degrees(np.arccos(ps/(n1*n2)))

#%%
# Définition des chemins
chemin_MOCAP = r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\MOCAP"

donnee_ant = pd.read_excel(
    r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\donnee_direct\donnees_antropo.xlsx",
    sheet_name='Feuil1'
)

mom_imp = pd.read_excel(
    r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\moment_impulsion.xlsx"
)

perf = pd.read_excel(r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\donnee_direct\perf.xlsx")

# Liste des sujets
sujets = mom_imp.iloc[:, 0].tolist()
print(sujets)

# Liste pour stocker les résultats de chaque fichier
resultats = []

# Boucle sur les fichiers MOCAP
for fichier in os.listdir(chemin_MOCAP):
    if fichier.endswith(".c3d"):
        chemin_fichier = os.path.join(chemin_MOCAP, fichier)

        # Extraire sujet, trial, répétition
        match = re.match(r"(.+?)_(.+?)_(.+?)\.c3d", fichier)
        if match:
            sujet = match.group(1)
            trial = match.group(2)
            repet = match.group(3)
            essai = f"{trial}{repet}"

            print(f"Lecture de {fichier} | Sujet: {sujet} | Trial: {trial} | Repetition: {repet} | Essai: {essai}")

            # Lecture du fichier c3d avec pyomeca
            rep = py.from_c3d(chemin_fichier)
            rep_mklist = np.array(rep.channel)  # liste des marqueurs
            temps_capture = (rep.attrs['last_frame'] - rep.attrs['first_frame']) / rep.attrs['rate']
            fréquence_mocap = rep.attrs['rate']
            time = np.array(rep.time)


            print(f"Temps de capture : {temps_capture:.2f} sec")

            # Extraction du marqueur MBWT
            Mbwt = np.array(rep.sel(channel='MBWT', axis=['x', 'y', 'z']))
 
#%% 
           
            if donnee_ant.loc[donnee_ant['Sujet'] == sujet, 'Jambe_Saut'].values[0] == "D" :
                Rkni = get_marker(rep, 'RKNI')
                Rani = get_marker(rep, 'RANI')
                Rkne = get_marker(rep, 'RKNE')
                Rane = get_marker(rep, 'RANE')
                Rfwt = get_marker(rep, 'RFWT')


                if Rane is not None and Rani is not None:
                    cheville = (Rane + Rani) / 2
                elif Rane is not None:
                    cheville = Rane
                elif Rani is not None:
                    cheville = Rani
                else:
                    cheville = None

                if Rkne is not None and Rkni is not None:
                    genoux = (Rkne + Rkni) / 2
                elif Rkne is not None:
                    genoux = Rkne
                elif Rkni is not None:
                    genoux = Rkni
                else:
                    genoux = None
                    
                hanche = Rfwt if Rfwt is not None else None
                
                
            else:
                Lkni = get_marker(rep, 'LKNI')
                Lani = get_marker(rep, 'LANI')
                Lkne = get_marker(rep, 'LKNE')
                Lane = get_marker(rep, 'LANE')
                Lfwt = get_marker(rep, 'LFWT')


                if Lane is not None and Lani is not None:
                    cheville = (Lane + Lani) / 2
                elif Lane is not None:
                    cheville = Lane
                elif Lani is not None:
                    cheville = Lani
                else:
                    cheville = None

                if Lkne is not None and Lkni is not None:
                    genoux = (Lkne + Lkni) / 2
                elif Lkne is not None:
                    genoux = Lkne
                elif Lkni is not None:
                    genoux = Lkni
                else:
                    genoux = None
                    
                hanche = Lfwt if Lfwt is not None else None
            
            cuisse =  np.where(np.isnan(genoux) | np.isnan(hanche), np.nan, genoux - hanche)
            tibia =  np.where(np.isnan(cheville) | np.isnan(genoux), np.nan, cheville - genoux)
            
            angle_genoux = angle_degre(cuisse, tibia)
            angle_genoux = 180 - angle_genoux
            

#%%

            mom_TD = mom_imp.loc[(mom_imp['sujet'] == sujet) & (mom_imp['essai'] == essai), 'TD'].values[0]
            mom_TO = mom_imp.loc[(mom_imp['sujet'] == sujet) & (mom_imp['essai'] == essai), 'TO'].values[0]
            print(mom_TO)
            print(f"TD: {mom_TD} | TO : {mom_TO}")

#%%            
            angle_genoux_impulsion = angle_genoux[mom_TD:mom_TO]
            flexion_max_genoux = np.nanmin(angle_genoux_impulsion)
            
            print(f"Flexion max genoux : {flexion_max_genoux}")
            
#%%
            # dist_moy_cheville = np.nanmean(cheville[0, mom_TD :mom_TO])
            # distance_planche = dist_moy_cheville / 10
            
            distance_planche = cheville[0, mom_TD]
            distance_planche = distance_planche / 10
            
            print(f"Planche à {distance_planche}cm")
            

#%%
            
            COM_2 = Mbwt         
            vit_COM_2 = abs(np.diff(COM_2)* fréquence_mocap)
            vit_COM_2 = vit_COM_2 / 1000       
            vit_COM_2_totale = np.sqrt(vit_COM_2[0]**2 + vit_COM_2[1]**2 + vit_COM_2[2]**2)
            
            vit_approche = np.nanmean(vit_COM_2[0, :mom_TD])  # Moyenne avant mom_TD (ligne 0)
            vit_saut_h = np.nanmean(vit_COM_2[0, mom_TO:])    # Moyenne après mom_TO (ligne 0)
            vit_saut_v = np.nanmean(vit_COM_2[2, mom_TO:])    # Moyenne après mom_TO (ligne 2)

            print(f"vit_approche: {vit_approche}| vit_saut_h: {vit_saut_h} | vit_saut_v: {vit_saut_v}")
            
            performance = perf.loc[perf['Sujet'] == sujet, essai].iloc[0]
            
            # À chaque itération, stocke les données sous forme de dictionnaire
            resultats.append({
                'Sujet': sujet,
                'Essai': essai,
                'vit_approche': vit_approche,
                'vit_saut_h': vit_saut_h,
                'vit_saut_v': vit_saut_v,
                'MKF': flexion_max_genoux,
                'distance planche': distance_planche,
                'perf': performance,
                })
            

              # Créer le nom du fichier CSV pour chaque sujet
nom_fichier = f"tableau_resultats.csv"

donnees_sujet = resultats
colonnes = list(donnees_sujet[0].keys())

            # Écrire les données dans un fichier CSV
with open(nom_fichier, "w", newline="") as fichier:
    writer = csv.DictWriter(fichier, fieldnames=colonnes)
    writer.writeheader()  # Écrire l'en-tête
    writer.writerows(donnees_sujet)  # Écrire toutes les lignes du tableau
        
        
