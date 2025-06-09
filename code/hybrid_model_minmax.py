from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from data import read_preprocessed_data
from model_visualisation import plot_confusion_matrix, plot_shap

# Opis modelu hybrydowego
# Model hybrydowy wykorzystuje klasyfikator typu "VotingClassifier" oparty na głosowaniu miękkim (soft voting),
# co oznacza, że bierze pod uwagę prawdopodobieństwa klasy z każdego modelu bazowego i oblicza średnią ważoną.
# Składa się z czterech różnych klasyfikatorów:
# 1. RandomForestClassifier – klasyfikator zespołowy oparty na wielu drzewach decyzyjnych.
# 2. KNeighborsClassifier – klasyfikator oparty na najbliższych sąsiadach (metoda leniwa).
# 3. LinearDiscriminantAnalysis – liniowy klasyfikator dyskryminacyjny, dobrze sprawdza się przy dobrze rozdzielonych klasach.
# 4. XGBClassifier – wydajny model oparty na gradient boosting, znany z wysokiej skuteczności w klasyfikacji wieloklasowej.
#
# Dane wejściowe są przetwarzane przy użyciu ColumnTransformer:
# - Zmienne numeryczne są standaryzowane za pomocą StandardScaler.
# - Zmienne kategoryczne są przepuszczane bez zmian.
#
# Dane są dzielone na zbiór treningowy i testowy w sposób stratyfikowany (z zachowaniem rozkładu klas).
# Model jest oceniany za pomocą walidacji krzyżowej (StratifiedKFold, 5 podziałów).
# Po dopasowaniu modelu, ocena skuteczności obejmuje dokładność, macierz pomyłek oraz metryki klasyfikacyjne.

# Wczytujemy dane przeskalowane za pomocą minmax scaler
X, Y = read_preprocessed_data(scaling_method='minmax')

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

# Modele bazowe
clf1 = ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
clf2 = ("knn", KNeighborsClassifier(n_neighbors=5))
clf3 = ("lda", LinearDiscriminantAnalysis())
clf4 = ("xgb", XGBClassifier(eval_metric="mlogloss", random_state=42))

# Voting Classifier – miękkie głosowanie
voting_clf = VotingClassifier(estimators=[clf1, clf2, clf3, clf4], voting="soft")

# Walidacja krzyżowa
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(voting_clf, X, Y, cv=cv, scoring="accuracy")

print("Wyniki walidacji krzyżowej")
print(f"Średnia dokładność: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
print(f"Wyniki foldów: {scores}")

# Trenowanie i testowanie
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print("\nMacierz pomyłek")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(voting_clf, X_test, y_test)

print("\nRaport klasyfikacji")
print(classification_report(y_test, y_pred))

print("\nAccuracy score")
print(accuracy_score(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average="macro")
print(f"\nF1-score (macro): {f1:.4f}")

# Obliczamy prawdopodobieństwa dla ROC AUC
probs = voting_clf.predict_proba(X_test)

# ROC AUC - wieloklasowe (one-vs-rest)
roc_auc_over = roc_auc_score(y_test, probs, multi_class="ovr")
print(f"ROC AUC (ovr): {roc_auc_over:.4f}")

# Tworzymy wykres SHAP pokazujący na ile dane zmienne wpłynęły na model 
# (zakomentowane, ponieważ zajmnuje dużo czasu output jest w folderze plots)
#plot_shap(voting_clf, X_test,"hybrid_minmax")

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