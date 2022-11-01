#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns



# In[2]:


data = pd.read_csv('boston_housing.csv')


# In[3]:


data


# Nettoyage des données

# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# Les variables zn, chas sont des variables constantes ou presque constante
# 
# 

# In[6]:


data_transform = data.drop(["zn", "chas"], axis = 1)


# Selection des variables pertinentes:

# In[7]:


prices = data_transform['medv']
Features = data_transform.drop('medv', axis = 1)
Features


# #### Afin d'ajuster un modèle de régression linéaire, nous sélectionnons les caractéristiques qui ont une forte corrélation avec notre variable cible 'medv'

# In[8]:


plt.figure(figsize=(20, 5))

features = ['crim', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
target = data_transform['medv']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = data_transform[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# #### On voit bien que les variables présentant une forte correlation avec la variable cible "medv" sont: 
# #### rm, lstat,  ptratio

# In[9]:


# On garde donc les variables rm, lstat, ptratio comme l'ensemble des variables 
# permettant de développer le modèle de regression

data_transform = data_transform[["rm", "lstat", "ptratio", "medv"]]


# ### Matrice de correlation

# In[10]:


sns.heatmap(data_transform.corr(), center = 0, cmap = 'RdBu', linewidths = 1, 
            annot = True, fmt = ".2f", vmin = -1, vmax = 1)
plt.title("Carte de correlation entre les variables pertinentes \n")


# On voit bien que la carte de correlation confirme bien la forte correlation des trois variables ("rm", "lstat", "ptratio") avec la variable cible "medv"

# ### Développement du model de regression

# In[11]:


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

prices = data_transform['medv']
features = data_transform.drop('medv', axis = 1)

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)


# ### Evaluons le  modèle de regression lineaire  à l'aide du cross validation

# In[12]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression

Lin_reg_model = LinearRegression()


scores_Linreg = cross_val_score(Lin_reg_model, features, prices, cv = 5)


print("Moyenne de cross-validation score: {: .2f}".format(scores_Linreg.mean()))


# ###  Entraînement et test du modèle

# In[13]:



from sklearn.metrics import mean_squared_error


Lin_reg_model.fit(X_train,  y_train)


# In[14]:


#Afficher les coefficients
print(Lin_reg_model.intercept_)
print(Lin_reg_model.coef_)


# In[18]:


#Afficher l'equation
cols_predicteurs = ["rm", "lstat", "ptratio"]
list(zip(cols_predicteurs, Lin_reg_model.coef_))


# ### Évaluation du modèle
# ### Nous évaluerons notre modèle en utilisant le RMSE et le score R2.

# In[19]:


# model evaluation for training set
y_train_predict = Lin_reg_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)



print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = Lin_reg_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

#comparer les valeurs test et prédites
test_pred_df = pd.DataFrame( { 'Valeurs test': y_test,
                'Valeurs prédites': np.round( y_test_predict, 2),
                'residuels': y_test - y_test_predict } )

print("Comparaison des valeurs tests et des valeurs prédites: ")

print("-----------------------------------------------")
print(test_pred_df[0:10])

print("The model performance for testing set")
print("--------------------------------------")
print("RMSE is : " + str(round(rmse, 2)))
print("R2 score is : " + str(round(r2, 2)))


# In[21]:


# Écriture des scores dans un fichier

with open("metrics.txt", 'w') as outfile:
        outfile.write("MSE:  {0:2.1f} \n".format(rmse))
        outfile.write("R2: {0:2.1f}\n".format(r2))


# ## Sauvegarde du modèle de regression pour l'utilisation dans le processus du CI

# In[16]:


import joblib


# In[17]:


joblib.dump(Lin_reg_model, 'Lin_reg_model.pkl')


# In[ ]:




