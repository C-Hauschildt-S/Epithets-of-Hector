import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Segoe UI"

df = pd.read_csv("hector_morphosyntax_v2.csv")

lemmata = [
    "Πριαμίδης",
    "Πρίαμος παῖς",
    "υἱός Πρίαμος",
    "υἱός Πρίαμος δαίφρων",
]

all_books = range(1, 25)

import numpy as np

n_books = len(all_books)
n_lemmata = len(lemmata)
x = np.arange(n_books)
width = 0.8 / n_lemmata
colors = ["#4a7c99", "#c0392b", "#27ae60", "#8e44ad"]

fig, ax = plt.subplots(figsize=(14, 5))

for i, lemma in enumerate(lemmata):
    subset = df[df["Lemma"] == lemma]
    counts = subset.groupby("Book").size().reindex(all_books, fill_value=0)
    offset = (i - n_lemmata / 2 + 0.5) * width
    ax.bar(x + offset, counts.values, width, label=lemma, color=colors[i])

ax.set_xlabel("Sang")
ax.set_ylabel("Antal forekomster")
ax.set_title("Patronymiske betegnelser for Hektor pr. sang")
ax.set_xticks(x)
ax.set_xticklabels(list(all_books))
ax.yaxis.get_major_locator().set_params(integer=True)
ax.legend()
plt.tight_layout()
fig.savefig("chart_patronymics_samlet.png", dpi=200)
plt.close(fig)
print("Gemt: chart_patronymics_samlet.png")