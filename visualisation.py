import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("18k_features_1_17.csv")
# sns.displot(penguins, x="flipper_length_mm", hue="species", multiple="stack")

ax = sns.displot(df, x="feature_6", hue="genre", multiple="stack")
ax.set(xlabel="Valor do Atributo (quantidade de versos únicos) para cada agrupamento de gênero", ylabel="Número de músicas")
plt.show()