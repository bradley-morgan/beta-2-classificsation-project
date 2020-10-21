import pandas as pd
import os
import matplotlib.pyplot as plt

# ag = 1 ant = 0
save = False
labels = {'ag': 1, 'ant': 0}

data_folders = os.listdir('../Data')
data_folders.remove('.DS_Store')

for sub_folder in data_folders:
    ag_data = pd.read_csv(os.path.join('../Data', sub_folder, 'ag.csv'))
    ant_data = pd.read_csv(os.path.join('../Data', sub_folder, 'ant.csv'))

    # Add Class label column
    ag_data["target"] = pd.Series([labels['ag']]*len(ag_data))
    ant_data["target"] = pd.Series([labels['ant']]*len(ant_data))

    ag_features = list(ag_data.columns.values)
    ant_features = list(ant_data.columns.values)

    combined_features = pd.Series(ag_features + ant_features)
    combined_features.value_counts().hist(bins=3)
    plt.xticks([1, 2])
    plt.yticks([25 * i for i in range(13)])
    plt.xlabel('Number of Shared features')
    plt.ylabel('Frequency')
    plt.title(f'PDB={sub_folder}: Shared Features Antagonist & Agonist DataFrames')

    shared_features = set(ag_features) & set(ant_features)

    # Combine ag and ant dataFrames and drop columns that NaN as
    # ant ligands contain observations atomic interactions that ag ligands do not and vice versa
    # Im not sure of any sensible way to perform imputation on these missing values yet.
    data = ag_data.append(ant_data, sort=False)
    data.dropna(axis=1, inplace=True)

    # save as csv
    if save:
        data.to_csv(os.path.join('../Data', sub_folder, 'data.csv'), index=False)
