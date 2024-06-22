import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score

def carga_csv(df):
    data = pd.read_csv(df)
    return data.head()
def modelo_cat_tree(X,y):

    clf_cat = DecisionTreeClassifier() #1 Defino el modelo y las variables

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42) #1.1 Divido los datos en train y test
    clf_cat = clf_cat.fit(X_train,y_train)

    '''#GridSearch

    params={'max_depth': [5],          #2 Defino los hiperparametros a probar en el gridsearch
            'max_leaf_nodes': [10,15,20], 
            'min_impurity_decrease' : [0.01, 0.02],
            'min_samples_split': [10,20], 
            'ccp_alpha': [0.0,0.01]
            }

    scoring = ['accuracy', 'roc_auc']  # 3 Defino las metricas a usar en el gridsearch

    grid_solver = GridSearchCV(estimator = clf_cat, # 4 Le digo a GridSearch todo lo que debe combinar
                    param_grid = params,
                    scoring = scoring,
                    cv = 5,
                    refit = 'roc_auc',
                    verbose = 2)

    model_result = grid_solver.fit(X_train,y_train) # 5 Entreno a GridSearch con los datos de train


    print("Mejor score:", model_result.best_score_) #imprimirá la mejor puntuación obtenida (roc_auc)
    print("Mejores parametros:",model_result.best_params_)'''


    #Predicciones test
    yhat = clf_cat.predict(X_test)
    y_probs = clf_cat.predict_proba(X_test)

    #Predicciones train
    yhat_train=clf_cat.predict(X_train)
    y_probs_train= clf_cat.predict_proba(X_train)

    return yhat_train, y_probs_train, yhat, y_probs
def cat_dummies(df):

    df = df.astype(str)
    data_cat_dum = pd.get_dummies(df, drop_first=True, dtype=int, sparse=True)
def importances(X,model):





    importances=pd.DataFrame([X.columns,model.feature_importances_], index=["feature","importance"]).T
    print(importances.sort_values("importance", ascending = False).head(10))
    return