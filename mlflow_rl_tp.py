#!/usr/bin/env python
# coding: utf-8

# In[2]:


# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
#! pip install --user mlflow
 
import os
import warnings
import sys
 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
 
import mlflow
import mlflow.sklearn
 
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
 
 

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
 
 
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
 
    # Read the wine-quality csv file from the URL
    #csv_url =\
        #'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        data = pd.read_csv('boston_housing.csv')
        data = data[["rm", "lstat", "ptratio", "medv"]]
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, may be the file not exists", e)
        
    # D'après le traitement des données, on peut noter qu'on a 3 prédicteurs: "rm", "lstat", "ptratio" et la
  
    
    #
 
    # Split the data into training and test sets. (0.75, 0.25) split.
    
    train, test = train_test_split(data)
 
     # la variable cible est: "medv" 
    
    train_x = train.drop(["medv"], axis=1)
    test_x = test.drop(["medv"], axis=1)
    train_y = train[["medv"]]
    test_y = test[["medv"]]
 
    #alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    #l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    alpha = 0.7 
    l1_ratio = 0.4

    mlflow.set_experiment(experiment_name="experiment3")
    mlflow.set_tracking_uri("http://localhost:5000") 
 
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
 
        predicted_qualities = lr.predict(test_x)
 
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
 
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
 
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
 
        mlflow.sklearn.log_model(lr, "model")
    
        #Sauvegarder le modele
        import pickle #une autre strategie
        pickle.dump(lr,open("c:\\tmp\\server_dev\\lr.dat","wb"))


# In[ ]:




