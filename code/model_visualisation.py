import pandas as pd
import numpy as np
import seaborn as sns
from data import read_preprocessed_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap

def plot_shap(model,X_test,name):
    explainer = shap.Explainer(model.predict_proba, X_test)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("plots/shap_summary_"+name+".png", bbox_inches='tight', dpi=300)
    plt.close()

'''
def plot_importances(model):
    importances = model.feature_importances_
    features = model.feature_names_in_

    forest_importances = pd.Series(importances, index=features)

    plt.figure(figsize=(10, 6))
    forest_importances.sort_values().plot(kind='barh')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

def plot_lda_coefficients(model):
    features = model.feature_names_in_
    coefs = model.coef_  # shape: (n_classes - 1, n_features)
    df = pd.DataFrame(coefs.T, index=features, columns=[f'LD{i+1}' for i in range(coefs.shape[0])])

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='coolwarm', center=0)
    plt.title('LDA Coefficient Heatmap')
    plt.ylabel('Features')
    plt.xlabel('Linear Discriminants')
    plt.tight_layout()
    plt.show()
'''

def plot_confusion_matrix(model, test_X, test_y):
    predict_y = model.predict(test_X)
    cost_labels = ["low cost","medium cost","high cost","very high cost"]
    cm = confusion_matrix(test_y, predict_y, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=cost_labels)
    disp.plot()
    plt.show()