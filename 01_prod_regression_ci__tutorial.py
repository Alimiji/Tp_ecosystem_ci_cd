#!/usr/bin/env python
# coding: utf-8

# # Tutoriel complet Regression lineaire
# ## Utilisation de l'intégration continue

# ## Collect data using pandas

# In[59]:


# modules nécessaires pour le notebook
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn import metrics
 


# In[60]:


# lire le fichier de données
#utiliser le param index_col: Column to use as the row labels of the DataFrame
df = pd.read_csv('Advertising.csv', 
                 index_col=0)
df.head()


# In[61]:


df.describe()


# # identification des descripteurs, cible et observations

# Quels sont les descripteurs? On a 3 descripteurs dans ce dataset qui sont:

# * TV
# * Radio
# * Newspaper

# Quelle est la cible?

# * Sales: vente d'un produit

# Quelle est la forme ou shape du dataframe?

# In[62]:


df.shape


# On voit que l'on a 200 observations avec 4 colonnes dont 3 sont des descripteurs

# # Tracé des relations entre les descripteurs et la cible

# In[63]:


#utilisation d'une figure avec 3 plots aligné sur une ligne
fig, axes = plt.subplots(1,3,sharey=False)
df.plot(kind='scatter', x='TV', y='sales', 
        ax=axes[0], figsize=(16,8))
df.plot(kind='scatter', x='radio', y='sales', 
        ax=axes[1], figsize=(16,8))
df.plot(kind='scatter', x='newspaper', y='sales', 
        ax=axes[2], figsize=(16,8))


# On voit au niveau des graphes qu'il existe une certaine relation linéaire entre TV et Sales ainsi que radio et Sales

# In[64]:


#meme chose mais avec seaborn

sns.pairplot(data=df, x_vars=['TV','radio','newspaper'], 
             y_vars='sales', height=7, aspect=0.7)


# # Tracé des correlations entre les différents descripteurs et cible

# * Cette partie n'a pas encore été faite.

# # Développement du modele linear regression

# In[65]:



cols_predicteurs = ['TV','radio','newspaper']
#predicteurs
X = df[cols_predicteurs]
y = df.sales


# In[66]:


#Effectuer la séparation Training-Test
 
 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, 
                                        y , test_size = 0.2, random_state=42)
#detail de chacun des sous-dataset
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[67]:


#estimation des coeeficients du modele lineaire
lm = LinearRegression()
lm.fit(X_train,y_train)
#Afficher les coefficients
print(lm.intercept_)
print(lm.coef_)


# In[68]:


#Afficher l'equation
list(zip(cols_predicteurs, lm.coef_))


# In[69]:


# proceder au test
y_pred = lm.predict(X_test)


# In[70]:


import numpy as np
#comparer les valeurs test et prédites
test_pred_df = pd.DataFrame( { 'Valeurs test': y_test,
                'Valeurs prédites': np.round( y_pred, 2),
                'residuels': y_test - y_pred } )
test_pred_df[0:10]


# In[71]:




# RMSE
mse = np.sqrt(metrics.mean_squared_error(y_test,
                                        y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,
                                        y_pred)))

#Calcul du R-squared
r2 = metrics.r2_score(y_test, y_pred)
print(r2)


# In[72]:


# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("MSE:  {0:2.1f} \n".format(mse))
        outfile.write("R2: {0:2.1f}\n".format(r2))


# In[73]:


#Référence: The Elements of Statistical Learning - Hastie, Tibshirani and Friedman, voir https://web.stanford.edu/~hastie/ElemStatLearn/

