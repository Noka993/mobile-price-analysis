import matplotlib.pyplot as plt
import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
file_path = os.path.join(root_dir, "data", "train.csv")
print(f"Wczytuję dane z: {file_path}")

df = pd.read_csv(file_path)

print(df.head())

# Kolorowa paleta
color_palette = ["#3a86ff", "#ff006e", "#8338ec"]

# Wybieramy kolumny z max 3 unikalnymi wartościami (binarne i 3-wartościowe)
bin_columns = [col for col in df.columns if df[col].nunique() <= 3]
print("Kolumny binarne i 3-wartościowe:", bin_columns)

# Wykresy kołowe
ax = (
    df[bin_columns]
    .apply(lambda x: x.value_counts())
    .plot(
        kind="pie",
        subplots=True,
        figsize=(10, 8),
        layout=(2, 3),
        colors=color_palette,
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=90,
        wedgeprops={'edgecolor': 'white'},
        legend=False,
    )
)
plt.tight_layout()
plt.show()

# Wykresy słupkowe
ax = (
    df[bin_columns]
    .apply(lambda x: x.value_counts())
    .plot(
        kind="bar",
        subplots=True,
        figsize=(10, 8),
        layout=(2, 3),
        color=color_palette,
        edgecolor='black',
        legend=False,
    )
)
plt.tight_layout()
plt.show()
