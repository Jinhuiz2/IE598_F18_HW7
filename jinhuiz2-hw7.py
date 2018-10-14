import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']


# Splitting the data into 70% training and 30% test subsets.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y,
                                                    random_state=0)

in_sample = []
out_sample = []
for i in range (1,15):
    rf = RandomForestClassifier(n_estimators=i,
                random_state=2)
    rf.fit(X_train, y_train) 
    y_pred_out = rf.predict(X_test)
    y_pred_in = rf.predict(X_train)
    out_sample_score = accuracy_score(y_test, y_pred_out)
    in_sample_score = accuracy_score(y_train, y_pred_in)
    in_sample.append(in_sample_score)
    out_sample.append(out_sample_score)
    print('n_estimators: %d, in_sample: %.3f, out_sample: %.3f'%(i, in_sample_score,
                                                             out_sample_score))


feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=13,random_state=2)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')

plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
