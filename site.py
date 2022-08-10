
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np

import math

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

from sklearn import tree, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from PIL import Image


  
st.title("Application pour le prix hypothétique d'une maison") 

sqft_living=st.number_input("Entrer la surface de l'espace de vie intérieur")
grade=st.number_input(' Entrer le niveau de qualité de la construction ')
sqft_above=st.number_input("Entrer la surface de l'espace intérieur du logement qui est au-dessus du niveau du sol")
sqft_living15=st.number_input("Entrer la surface de l'espace de vie intérieur du logement pour les 15 voisins les plus proches")
bathrooms=st.number_input("Entrer le nombre de salles de bains, selon les conventions américaines")
view=st.number_input("Entrer un nombre représentant la qualité de la vue de 0 à 4")
sqft_basement=st.number_input("Entrer la surface de l'espace de dessous du niveau du sol")
bedrooms=st.number_input("Entrer le nombre de chambres")
lat=st.number_input('Entrer la lattitude de votre maison')
long=st.number_input('Entrer la longitude de votre maison')
waterfront=st.number_input(" Voulez vous une vue sur la mer ,0 pour  Oui  et 1 pour Non")
zipcode = st.number_input("Entrer le code postale de votre maison: ")



  
#créer une fonction user_input qui met les infos dans un dictionnaire puis transformer en dataframe , puis predire le prix via le modele
if(st.button('Calculer le prix hypothétique')): 
    data={
    'bedrooms':bedrooms,
    'bathrooms':bathrooms,            
    'sqft_living':sqft_living,
    'waterfront': waterfront,
    'view':view,        
    'grade':grade,
    'sqft_above':sqft_above,
    'sqft_basement':sqft_basement,
    'zipcode':zipcode,
    'lat':lat,
    'long':long,
    'sqft_living15':sqft_living15



    }
    maison_propriete=pd.DataFrame(data,index=[0])
    df= pd.read_csv("kc_house_data_entrainement.csv")
    df.drop([ "id"], axis=1, inplace=True)
    df.drop([ "floors"], axis=1, inplace=True)
    df.drop([ "yr_renovated"], axis=1, inplace=True)
    df.drop([ "yr_built"], axis=1, inplace=True)
    df.drop(["date"], axis=1, inplace=True)
    df.drop(["sqft_lot"], axis=1, inplace=True)
    df.drop(["condition"], axis=1, inplace=True)
    df.drop(["sqft_lot15"], axis=1, inplace=True)
    X = df.drop('price', axis=1)
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model1 = make_pipeline(StandardScaler(), Ridge())

    model1.fit(X_train, y_train)

    prediction = model1.predict(maison_propriete)

    prix = round(prediction[0])
    st.write("# Le prix de votre maison est:", prix, "$")
    