import pandas as pd
import numpy as np
from data import read_preprocessed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import plot_tree
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
X,y = read_preprocessed_data(scaling_method='standard')




train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=1000, max_depth=5,random_state=42)
clf.fit(train_X, train_y)

predict_y = clf.predict(test_X)

print("Accuracy score: ", clf.score(test_X, test_y))

# F1-score (macro średnia dla wieloklasowej klasyfikacji)
f1 = f1_score(test_y, predict_y, average='macro')
print("F1-score (macro):", f1)

# Prawdopodobieństwa predykcji dla AUC
probs = clf.predict_proba(test_X)

# ROC AUC (one-vs-rest) dla wieloklasowego problemu
auc = roc_auc_score(test_y, probs, multi_class='ovr')
print("ROC AUC (ovr):", auc)


'''
best_tree = None
best_score = 0

for i, tree in enumerate(clf.estimators_):
    score = tree.score(test_X.to_numpy(), test_y.to_numpy())
    if score > best_score:
        best_score = score
        best_tree = tree
        best_index = i

estimator = clf.estimators_[best_index] 
'''

estimator = clf.estimators_[0] 
plt.figure(figsize=(20, 10))
plot_tree(estimator, 
          feature_names=clf.feature_names_in_, 
          class_names=[str(cls) for cls in clf.classes_], 
          filled=True, 
          max_depth=4,
          impurity=False)
plt.show()

importances = clf.feature_importances_
features = clf.feature_names_in_

forest_importances = pd.Series(importances, index=features)

plt.figure(figsize=(10, 6))
forest_importances.sort_values().plot(kind='barh')
plt.title("Feature Importances")
plt.tight_layout()
plt.show()