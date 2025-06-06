import pandas as pd
import numpy as np
from sklearn.calibration import label_binarize
from data import read_preprocessed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model_visualisation import plot_shap, plot_confusion_matrix

X, y = read_preprocessed_data(scaling_method="standard")

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)
clf.fit(train_X, train_y)

predict_y = clf.predict(test_X)

print("Accuracy score: ", clf.score(test_X, test_y))

# F1-score (macro średnia dla wieloklasowej klasyfikacji)
f1 = f1_score(test_y, predict_y, average="macro")
print("F1-score (macro):", f1)

# Prawdopodobieństwa predykcji dla AUC
probs = clf.predict_proba(test_X)

# ROC AUC (one-vs-rest) dla wieloklasowego problemu
roc_auc_over = roc_auc_score(test_y, probs, multi_class="ovr")
print("ROC AUC (ovr):", roc_auc_over)

best_tree = None
best_score = 0

for i, tree in enumerate(clf.estimators_):
    score = tree.score(test_X.to_numpy(), test_y.to_numpy())
    if score > best_score:
        best_score = score
        best_tree = tree
        best_index = i

estimator = clf.estimators_[best_index]

# estimator = clf.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(estimator, 
        feature_names=clf.feature_names_in_, 
        class_names=[str(cls) for cls in clf.classes_], 
        filled=True, 
        max_depth=3,
        impurity=False,
        label='root')
plt.savefig('plots/random_forest_tree.png', dpi=300, bbox_inches='tight')
plt.show()

plot_confusion_matrix(clf, test_X, test_y)

# Tworzymy wykres SHAP pokazujący na ile dane zmienne wpłynęły na model
# (zakomentowane, ponieważ zajmnuje bardzo dużo czasu output jest w folderze plots)
# plot_shap(clf, test_X, "random_forest")

n_classes = 4
y_test_bin = label_binarize(test_y, classes=np.arange(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ["blue", "green", "orange", "red"]
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        color=colors[i],
        lw=2,
        label=f"Klasa {i} (AUC = {roc_auc[i]:.2f})",
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Krzywe ROC – VotingClassifier (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
