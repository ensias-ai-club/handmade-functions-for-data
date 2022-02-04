# Copy Paste Hub for Data Scientists

> In progress

There is alot of functions and commands that you would use frequently when analyzing data
in this repo we provide some functions that you would use to save time

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


