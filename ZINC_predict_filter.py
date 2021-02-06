from tools.Data import Preprocess
from tools.make_models import make_model
import Config_Script
import wandb
import pandas as pd
import numpy as np
import tools.quick_tools as q_tools

# TODO Process Ligand Poses

wandb.init()
meta_data = Config_Script.get_config().feature_importance_config
meta_data.load_model_from = 'cloud'

train_data = Preprocess.load_from_local('./data/processed/filtered/filter-train-processor_98.pickle')
# full_x_train = np.concatenate((train_data.x_train, train_data.x_hold_out), axis=0)
# full_y_train = np.concatenate((train_data.y_train, train_data.y_hold_out), axis=0)

zinc_data = Preprocess.load_from_local('./data/processed/zinc/filter-zinc-train-processor_98.pickle')

model = make_model(meta_data.model)(meta_data)
# model.fit(full_x_train, full_y_train)

# reduce zinc data to shared features
zinc_data.data = pd.DataFrame(zinc_data.data, columns=zinc_data.feature_names)
zinc_data.data = zinc_data.data[train_data.feature_names]
zinc_data.feature_names = zinc_data.data.columns.to_numpy()
zinc_data.removed_features = q_tools.process_ligand_poses(zinc_data.removed_features.to_numpy()[:, 0])

# Generalise to ZINC
zinc_y_preds = model.predict(zinc_data.data)
zinc_y_proba = model.predict_proba(zinc_data.data)

# Group predictions across ligand poses
avg_zinc_groups = pd.DataFrame(
zip(zinc_data.removed_features, zinc_y_preds), columns=['ligand_pose', 'y_preds']
).groupby(['ligand_pose'])

# convert predictions to text representations
output = pd.DataFrame(zip(zinc_y_preds, zinc_y_proba[:, 1]), columns=['prediction', 'Agonist probability'])
excel_output = pd.read_csv('./data/filter_correlation/correlation2.csv')
excel_output.rename(columns={'FILTER': 'Correlation'}, inplace=True)
x = []
for j in range(len(excel_output)):
    p = excel_output['Correlation'].iat[j]
    if p >= 0.4:
        x.append('AGONIST')
    else:
        x.append('ANTAGONIST')
excel_output['Correlation'] = x

correlation_set = set(excel_output['ID'])
zinc_set = set(zinc_data.removed_features)
shared_ligand_poses = correlation_set.intersection(zinc_set)

# Extract only shared ligand poses between correlation and Zinc data
excel_output = excel_output[excel_output['ID'].isin(shared_ligand_poses)]

# Compute overlap
avg_rfc_preds = []
avg_rfc_probs = []
for shared_ligand in shared_ligand_poses:
    zinc_group = avg_zinc_groups.get_group(shared_ligand)
    # average prediction
    avg_proba = zinc_group['y_preds'].mean()
    avg_pred = zinc_group['y_preds'].mean().round()
    avg_pred = 'ANTAGONIST' if avg_pred == 0 else 'AGONIST'
    avg_rfc_preds.append(avg_pred)
    avg_rfc_probs.append(avg_proba)

excel_output['RFC'] = avg_rfc_preds
matches = excel_output['Correlation'] == excel_output['RFC']
true_count = matches.value_counts().loc[True]
overlap_percentage = np.round(true_count / len(excel_output) * 100)
print(f'Correlation vs RFC: Percentage of Matched Predictions = {overlap_percentage}%')

# Save data
rfc_out = pd.DataFrame(zip(list(shared_ligand_poses), avg_rfc_preds, avg_rfc_probs), columns=['ID', 'Prediction', 'Probability'])
rfc_out.to_csv('analysis/zinc_filter_RFC_98-46.csv', index=False)

