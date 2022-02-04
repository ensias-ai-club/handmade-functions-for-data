# Copy Paste Hub for Data Scientists

> In progress

There is alot of functions and commands that you would use frequently when analyzing data for example :
* Checking for missing values
* Scaling features
* Feature Correlations
* Cross Validation
* Evaluation of models

in this repo we provide some functions that you would use to save time.

> some of these functions require some tweaking according to the situation

## Checking for Missing Values

~~~python
import pandas as pd

(df.isnull().sum()*100 / df.shape[0]).sort_values(ascending = False).head(15).to_frame('nan_percent')
~~~
## Visualize Feature Distributions
~~~python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_dist(data):
    fig = plt.figure(figsize = (15, 60))
    for i in range(len(data.columns.tolist())):
        plt.subplot(20,5,i+1)
        sns.set_style("white")
        plt.title(data.columns.tolist()[i], size = 12, fontname = 'monospace')
        a = sns.kdeplot(data[data.columns.tolist()[i]], color = '#34675c', shade = True, alpha = 0.9, linewidth = 1.5, edgecolor = 'black')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks(fontname = 'monospace')
        plt.yticks([])
        for j in ['right', 'left', 'top']:
            a.spines[j].set_visible(False)
            a.spines['bottom'].set_linewidth(1.2)
    fig.tight_layout(h_pad = 3)
    plt.show()
~~~

## Scaling Features
~~~python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])
~~~

## Correlation Matrix
~~~python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data)
~~~
## Cross Validation
~~~python

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (tr_index , val_index) in enumerate(kf.split(train)):
    
    print("⁙" * 10)
    print(f"Fold {fold + 1}")
    
    x_train,x_val = x.values[tr_index] , x.values[val_index]
    y_train,y_val = y.values[tr_index] , y.values[val_index]
        
    eval_set = [(x_val, y_val)]
    
    model = LinearRegression()
    model.fit(x_train, y_train, eval_set = eval_set, verbose = False)
    
    train_preds = model.predict(x_train)
    val_preds = model.predict(x_val)
    
    print(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    if test_preds is None:
        test_preds = model.predict(test.values)
    else:
        test_preds += model.predict(test.values)

print("-" * 50)
print("\033[95mTraining Done")

test_preds /= 10
~~~

## Evaluation of classification models
~~~python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluation(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_valid)
    
    cm = confusion_matrix(y_valid, ypred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
    
    print(classification_report(y_valid, ypred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='accuracy',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.show()
~~~

## Hyperparametres optimization

~~~python
import optuna
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

def objective(trial):
    # define the models and the parametres spaces for each model    
    classifier_name = trial.suggest_categorical('classifier', ['DecisionTreeClassifier', 
                                                               'RandomForestClassifier', 
                                                               'LogisticRegression', 
                                                               'KNeighborsClassifier',
                                                               'XGBClassifier'])
    if classifier_name == 'DecisionTreeClassifier':
        max_depth = trial.suggest_int("max_depth", 2, 612)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 612)
        max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 612))
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        classifier_obj = DecisionTreeClassifier(criterion = criterion, 
                                                max_depth = max_depth, 
                                                min_samples_split = min_samples_split,
                                                max_leaf_nodes = max_leaf_nodes)
    elif classifier_name == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        max_depth = trial.suggest_int('max_depth', 4, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 1, 150)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 60)
        classifier_obj = RandomForestClassifier(n_estimators = n_estimators,
                                                max_depth = max_depth,
                                                min_samples_split = min_samples_split,
                                                min_samples_leaf = min_samples_leaf)
    elif classifier_name == 'LogisticRegression':
        tol = trial.suggest_uniform('tol' , 1e-6 , 1e-3)
        C = trial.suggest_loguniform("C", 1e-2, 1)
        fit_intercept = trial.suggest_categorical('fit_intercept' , [True, False])
        solver = trial.suggest_categorical('solver' , ['lbfgs','liblinear'])
        classifier_obj = LogisticRegression(tol = tol, C = C, fit_intercept = fit_intercept, solver = solver)
    elif classifier_name == "XGBClassifier":
        params = {'objective' : 'binary:logistic',
              'learning_rate' : trial.suggest_float("learning_rate", 0.01, 0.8),
              'n_estimators' : trial.suggest_int("n_estimators", 100, 1000),
              'max_depth' : trial.suggest_int("max_depth", 2, 600),
              'min_child_weight' : trial.suggest_int("min_child_weight", 1, 20),
              'gamma' : trial.suggest_float("gamma", 0.1, 1),
              'subsample' : trial.suggest_float("subsample", 0.1, 1),
              'colsample_bytree' : trial.suggest_float("colsample_bytree", 0.1, 1)
             }
        classifier_obj = xgb.XGBClassifier(**params)
    else:
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        classifier_obj = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, metric = metric)
    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

# Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)
~~~

## Put a jupyter notebook in docker

~~~dockerfile
FROM python:3.9.1

EXPOSE 8888
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update
RUN pip install jupyter jupyterlab
RUN pip install requirements.txt

CMD [ "jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root" ]

~~~




