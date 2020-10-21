import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../../Data/B2in/data.csv')

X = data.drop(['target', 'Ligand_Pose'], axis=1)
Y = data.target

data["target"].value_counts().plot(kind="bar")
plt.xlabel('Classes Agonist(1) Antagnoist(0)')
plt.ylabel('Frequency')
plt.title(f'PDB=B2inThe Frequency of Classes in DataFrame')
plt.grid()
plt.show()