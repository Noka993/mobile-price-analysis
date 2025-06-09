from sklearn.calibration import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from data import read_preprocessed_data, outliers_statistics

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
import matplotlib.pyplot as plt

from model_visualisation import plot_confusion_matrix, plot_shap
X,Y= read_preprocessed_data(scaling_method='minmax')


# 2. Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
LogReg  = LogisticRegression(solver='liblinear', random_state=42)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(LogReg, X, Y, cv=cv, scoring='accuracy')

print("Wyniki walidacji krzyżowej:")
print(f"Średnia dokładność: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"Poszczególne wyniki: {cv_scores}")


print("Leave-One-Out (LOO) walidacja:")
loo = LeaveOneOut()
loo_scores = cross_val_score(LogReg, X, Y, cv=loo)

print(f"Liczba iteracji: {len(loo_scores)}")
print(f"Średnia dokładność: {np.mean(loo_scores):.4f}")



print("Podsumowanie:")
print(f"LOO dokładność: {np.mean(loo_scores):.4f}")



LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
#wstępna analiza
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(LogReg, X_test, y_test)
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))
print("\nDokładność:", accuracy_score(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nF1-score (macro): {f1:.4f}")

# Obliczamy prawdopodobieństwa dla ROC AUC
probs = LogReg.predict_proba(X_test)

# ROC AUC - wieloklasowe (one-vs-rest)
roc_auc_over = roc_auc_score(y_test, probs, multi_class="ovr")
print(f"ROC AUC (ovr): {roc_auc_over:.4f}")

# Tworzymy wykres SHAP pokazujący na ile dane zmienne wpłynęły na model
# (zakomentowane, ponieważ zajmnuje dużo czasu output jest w folderze plots)
#plot_shap(LogReg, X_test, "LogReg_minmax")

n_classes = 4
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

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