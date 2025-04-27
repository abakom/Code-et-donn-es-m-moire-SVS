# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:02:07 2025

@author: pc
"""

#%% Logiciels

from pyomeca import Markers as py
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
import xarray as xr
import csv
import sys
from scipy.signal import butter, filtfilt


#%% Fonctions
def Lire_fichier (fichier) : 
    df = pd.read_csv(fichier, sep=';')
    return df

def moment_debut_impulsion(moyenne_tot_quadri, mom_max_quadri, emg_data): # Déterminer a quel moment débute le TD via la contraction du quadriceps avant la contraction max
    col1 = emg_data.iloc[:, 1]  # On considère que la colonne 1 du tableau "emg_data" contient les valeurs de x    
    for i in range(mom_max_quadri -1, -1, -1):  # On parcourt la ligne 3 avant mom_max_quadri (donc jusqu'à mom_max_quadri - 1)# On part de mom_max_quadri-1 et on remonte
        x_value = col1[i]  # Valeur de x à l'indice i de la ligne 3
        if x_value <= 5 * moyenne_tot_quadri:  # Si x est inférieur à 5 * moyenne du test
            return i  # Retourner l'indice du moment trouvé
                

def moment_fin_impulsion(moyenne_tot_tibial, mom_max_quadri, emg_data):   # Pareil que début impulsion mais après
    col9 = emg_data.iloc[:, 9]  # On considère que la colonne 9 du tableau "emg_data" contient les valeurs de x    
    for i in range(mom_max_quadri, len(col9)):# On parcourt la colonne 9 après mom_max_quadri (donc à partir de mom_max_quadri)
        x_value = col9[i]
        if x_value <= 5 * moyenne_tot_tibial:  # Si x est supérieur à 5 * moyenne test
            return i 
        
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs  # Fréquence de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Fonction d'application du filtre
def apply_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def compute_rms(signal, window_size):
    rms_values = np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='same'))
    return rms_values


#%% Définir dossier

#chemin_EMG = r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\EMG\Mesure_2"
chemin_EMG = r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\EMG\Mesure_1"
#chemin_TEST = r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\TEST\Mesure_2"
chemin_TEST = r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\TEST\Mesure_1"

donnee_ant = pd.read_excel(r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\donnee_direct\donnees_antropo.xlsx", sheet_name='Feuil1') # Données spécifique au sauteur
perf = pd.read_excel(r"C:\Users\pc\Documents\Cours_ENS\memoire\SVS\Collecte donnees\test\donnee_direct\perf.xlsx")

sujets = donnee_ant.iloc[:, 0].tolist()  # Récupérer tous les sujets de la première colonne et les convertir en liste
noms_a_enlever = ['Lucien']

# Utiliser une boucle pour enlever chaque élément dans la liste des noms à enlever
for nom in noms_a_enlever:
    if nom in sujets:
        sujets.remove(nom)
print(sujets)

#%% Boucle sujet

fs = 1925  # Fréquence d'échantillonnage en Hz
lowcut = 40.0  # Fréquence de coupure basse en Hz
highcut = 400.0  # Fréquence de coupure haute en Hz
order = 2  # Ordre du filtre
window_size = 100  # Taille de la fenêtre pour le calcul de l'enveloppe RMS (par exemple 100 échantillons)



resultats_emg = []

for sujet in sujets: # boucle sujet 
    
    fichiers_test = [os.path.join(chemin_TEST, f) for f in os.listdir(chemin_TEST) if f.startswith(f"{sujet}_test")]
    fichiers_emg = [os.path.join(chemin_EMG, f) for f in os.listdir(chemin_EMG) if f.startswith(f"{sujet}_AP_") or f.startswith(f"{sujet}_SP_")]
    #fichiers_mocap = [os.path.join(chemin_MOCAP, f) for f in os.listdir(chemin_MOCAP) if f.startswith(f"{sujet}_AP_") or f.startswith(f"{sujet}_SP_")]
        
  #%% Fichier force max test EMG
  
    test_1 = []
    test_2 = []
   
    for fichier in fichiers_test : # boucle fichier dans les fichiers de test emg
       
       file_path = fichier
       
       nom_fichier = os.path.basename(file_path) #prendre uniquement le nom du fichier
       parts = nom_fichier.split('_')  # On divise le nom avec '_'
       sujet = parts[0]
       trial = parts[2]  
       repet = int(parts[3].split('.')[0])  # Enlever l'extension de fichier et convertir en entier
       
       globals()[nom_fichier] = Lire_fichier(file_path)
       df= Lire_fichier(file_path)
       
       # Remplacer toutes les virgules par des points dans toutes les cellules
       df = df.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)  
       
       
       T1 = df.iloc[1 , 0]
       T1 = float(T1)
       df.iloc[1:, 0] = df.iloc[1:, 0].astype(float) - T1
       
       #df.iloc[:, 1] = df.iloc[:, 1].rolling(window=10, min_periods=1).mean()
       
       # Nom de la colonne 1 (deuxième colonne)
       col_name = df.columns[1]

        # Conversion en float (très important avant filtrage)
       df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        # Appliquer le filtre Butterworth passe-bande
       df[col_name] = apply_bandpass_filter(df[col_name].values, lowcut, highcut, fs, order)
       
       # Calcul de l'enveloppe RMS pour chaque colonne
       rms_test = compute_rms(df[col_name].values, window_size)

       # Remplacer les signaux EMG par leurs enveloppes RMS pour analyses suivantes
       df[col_name] = rms_test

       
       if trial == "1" :
           test_1.append(df.iloc[:, 1].max()) 
       else : 
           test_2.append(df.iloc[:, 1].max())
           
    max_test_1 = max(test_1)
    #max_test_quadri = max_test_1.replace(',', '.')
    #max_test_quadri = float(max_test_quadri)
    max_test_quadri = float(max_test_1)
   
    max_test_2 = max(test_2)
    #max_test_tibial = max_test_2.replace(',', '.')
    #max_test_tibial = float(max_test_tibial)
    max_test_tibial = float(max_test_2)
    
    #%% EMG
    
    for fichier in fichiers_emg : 
        
        nom_fichier = os.path.basename(fichier) #prendre uniquement le nom du fichier
        parts = nom_fichier.split('_')  # On divise le nom avec '_'
        sujet = parts[0]
        trial = parts[1]  
        repet = int(parts[2].split('.')[0])  # Enlever l'extension de fichier et convertir en entier
             
        essai = f"{trial}{repet}"
        
        emg_data = Lire_fichier(fichier)
        
        emg_data = emg_data.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)  
        
        T1 = emg_data.iloc[1 , 0]
        T1 = float(T1)
        emg_data.iloc[1:, 0] = emg_data.iloc[1:, 0].astype(float) - T1
       
        
       # Nom de la colonne 1 (deuxième colonne)
        col_name_1 = emg_data.columns[1]
        col_name_2 = emg_data.columns[9]

        # Conversion en float (très important avant filtrage)
        emg_data[col_name_1] = pd.to_numeric(emg_data[col_name_1], errors='coerce')
        emg_data[col_name_2] = pd.to_numeric(emg_data[col_name_2], errors='coerce')

        # Appliquer le filtre Butterworth passe-bande
        emg_data[col_name_1] = apply_bandpass_filter(emg_data[col_name_1].values, lowcut, highcut, fs, order)
        emg_data[col_name_2] = apply_bandpass_filter(emg_data[col_name_2].values, lowcut, highcut, fs, order)
        
        
        # Calcul de l'enveloppe RMS pour chaque colonne
        rms_quadri = compute_rms(emg_data[col_name_1].values, window_size)
        rms_tibial = compute_rms(emg_data[col_name_2].values, window_size)

        # Remplacer les signaux EMG par leurs enveloppes RMS pour analyses suivantes
        emg_data[col_name_1] = rms_quadri
        emg_data[col_name_2] = rms_tibial
        
        fréquence_emg = 1925
        
        abs_emg_data_quadri = abs(emg_data.iloc[:, 1])
        abs_emg_data_tibial = abs(emg_data.iloc[:, 9])
        moyenne_tot_quadri = abs_emg_data_quadri.mean()
        moyenne_tot_tibial = abs_emg_data_tibial.mean()
        
        max_quadri_abs = emg_data.iloc[:, 1].max()#EMG quadri dans colonne 1
        max_quadri_abs = float(max_quadri_abs)
        max_quadri = (max_quadri_abs / max_test_quadri) * 100
        mom_max_quadri = emg_data.iloc[:, 1].idxmax()  # Utilisation de idxmax pour obtenir l'indice
        max_tibial_abs = emg_data.iloc[:, 9].max() # EMG tibial dans colonne 9
        max_tibial_abs = float(max_tibial_abs)
        max_tibial = (max_tibial_abs / max_test_tibial) * 100
        mom_max_tibial = emg_data.iloc[:, 9].idxmax()  # Utilisation de idxmax pour obtenir l'indice
        
        
        
        mom_TD_emg = moment_debut_impulsion(moyenne_tot_quadri, mom_max_quadri, emg_data)
        mom_TO_emg = moment_fin_impulsion(moyenne_tot_tibial, mom_max_quadri, emg_data)
        
        int_quadri_float = emg_data.iloc[:,1]
        int_quadri_float = int_quadri_float.astype(float)
        int_quadri_abs = np.sum(int_quadri_float.iloc[mom_TD_emg:mom_TO_emg])
        int_quadri = int_quadri_abs / max_test_quadri
        
        int_tibial_float = emg_data.iloc[:,9]
        int_tibial_float = int_tibial_float.astype(float)
        int_tibial_abs = np.sum(int_tibial_float.iloc[mom_TD_emg:mom_TO_emg])
        int_tibial = int_tibial_abs / max_test_tibial
        
        performance = perf.loc[perf['Sujet'] == sujet, essai].iloc[0]             
        
        resultats_emg.append({
            "Sujet": sujet,
            "Essai" : essai,
            "Perf" : performance,
            "Tibial max" : max_tibial,
            "Intégral tibial" : int_tibial,
            "Quadri max" : max_quadri,
            "Intégral quadri" : int_quadri,
            })
        

        
        
donnees_sujet = resultats_emg
colonnes = list(donnees_sujet[0].keys())
    
    # Créer le nom du fichier CSV pour chaque sujet
nom_fichier = f"tableau_emg_2.csv"
    
    # Écrire les données dans un fichier CSV
with open(nom_fichier, "w", newline="") as fichier:
    writer = csv.DictWriter(fichier, fieldnames=colonnes)
    writer.writeheader()  # Écrire l'en-tête
    writer.writerows(donnees_sujet)  # Écrire toutes les lignes du tableau

    print(f"\nLes données du sujet {sujet} ont été enregistrées dans '{nom_fichier}'.")
        