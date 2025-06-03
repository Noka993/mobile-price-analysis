# 📱 Mobile Phone Price Classification

This project aims to classify mobile phones into four price categories based on their technical specifications using various machine learning models.

## 📂 Project Structure

```
├── code/
│   ├── __init__.py
│   ├── data.py                      # Data loading utilities
│   ├── forest.py                    # Random Forest
│   ├── hybrid_model.py             # VotingClassifier (ensemble)
│   ├── LDA.py                       # Linear Discriminant Analysis
│   ├── LogReg.py                    # Logistic Regression
│   ├── model_visualisation.py      # SHAP plots and other model visuals
│   ├── statystyki.py               # Descriptive statistics
│   ├── transform.py                # Feature scaling and preprocessing
│   ├── visualization.py            # Data visualization utilities
├── data/
│   ├── test.csv                     # Test dataset
│   ├── train.csv                    # Training dataset
├── plots/                          # Output visualizations
├── .gitignore
├── LICENSE                         # License file
├── MAD_2.pdf                       # Project report (Polish)
├── README.md                       # You're here!
├── requirements.txt                # Required Python packages
```

## 🧠 Project Summary

This project investigates whether technical attributes of a phone (e.g., RAM, battery, screen resolution, presence of 4G, etc.) can accurately predict its price category. The data comes from the [Mobile Price Classification dataset on Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).

**Target variable:** `price_range` (0 = cheapest, 3 = most expensive)

## 🔍 Models Used

- **Logistic Regression**
- **Linear Discriminant Analysis (LDA)**
- **Random Forest**
- **Hybrid Ensemble Model** using `VotingClassifier` (Random Forest + KNN + LDA + XGBoost)

## 📊 Results

| Model              | Accuracy | F1-score | ROC AUC |
|-------------------|----------|----------|---------|
| Logistic Regression | 82%     | 0.8201   | 0.9394  |
| LDA                 | 79%     | 0.7910   | 0.9300  |
| Random Forest       | 91%     | 0.9097   | 0.9846  |
| Hybrid Ensemble     | 89%     | 0.8893   | 0.9780  |

## 🔧 Setup & Installation

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Code

Scripts are modularized under the `code/` directory. You can run experiments and analysis by executing:

```bash
python -m code.hybrid_model
```

For visualizations or statistics:

```bash
python -m code.visualization
python -m code.statystyki
```

## 📄 Report

For a comprehensive explanation of the project—including methodology, preprocessing, evaluation, and visualizations—refer to the [`MAD_2.pdf`](MAD_2.pdf) file.

## 🧾 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

**Keywords:** mobile classification, machine learning, logistic regression, LDA, Random Forest, ensemble learning, SHAP, data visualization
