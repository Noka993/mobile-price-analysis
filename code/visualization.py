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
bin_columns = [col for col in df.columns if df[col].nunique() <= 4]
nonbin_columns = [col for col in df.columns if col not in bin_columns]
print("Kolumny binarne i 3/4-wartościowe:", bin_columns)
axes = df[nonbin_columns].hist(
    figsize=(12, 10),
    layout=(4, 4),
    color=color_palette[0],
    edgecolor="black",
    grid=False,
)


for ax in axes.flatten():
    ax.tick_params(axis="both", labelsize=8)

plt.tight_layout()
plt.show()
# Wyświetlamy boxploty dla kolumn niebinarnych
ax = df[nonbin_columns].plot(
    kind="box",
    subplots=True,
    figsize=(12, 10),
    layout=(5, 5),
    color=color_palette[1],
)

plt.tight_layout()
plt.show()


print("Kolumny binarne i 3/4-wartościowe:", bin_columns)

# Wykresy kołowe
ax = (
    df[bin_columns]
    .apply(lambda x: x.value_counts())
    .plot(
        kind="pie",
        subplots=True,
        figsize=(10, 8),
        layout=(3, 3),
        colors=color_palette,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white"},
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
        layout=(3, 3),
        color=color_palette,
        edgecolor="black",
        legend=False,
    )
)
plt.tight_layout()
plt.show()
