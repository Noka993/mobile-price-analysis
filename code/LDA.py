import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, LeaveOneOut,  StratifiedKFold
from data import read_preprocessed_data,outliers_statistics
from model_visualisation import plot_confusion_matrix, plot_shap

X,Y= read_preprocessed_data(scaling_method='standard')
# 2. Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
lda=LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lda, X, Y, cv=cv, scoring='accuracy')

print("Wyniki walidacji krzyżowej:")
print(f"Średnia dokładność: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"Poszczególne wyniki: {cv_scores}")


print("Leave-One-Out (LOO) walidacja:")
loo = LeaveOneOut()
loo_scores = cross_val_score(lda, X, Y, cv=loo)

print(f"Liczba iteracji: {len(loo_scores)}")
print(f"Średnia dokładność: {np.mean(loo_scores):.4f}")



print("Podsumowanie:")
print(f"LOO dokładność: {np.mean(loo_scores):.4f}")



lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
#wstępna analiza
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(lda, X_test, y_test)
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))
print("\nDokładność:", accuracy_score(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nF1-score (macro): {f1:.4f}")

# Obliczamy prawdopodobieństwa dla ROC AUC
probs = lda.predict_proba(X_test)

# ROC AUC - wieloklasowe (one-vs-rest)
auc = roc_auc_score(y_test, probs, multi_class='ovr')
print(f"ROC AUC (ovr): {auc:.4f}")

# Tworzymy wykres SHAP pokazujący na ile dane zmienne wpłynęły na model 
# (zakomentowane, ponieważ zajmnuje dużo czasu output jest w folderze plots)
#plot_shap(lda, X_test, "LDA")