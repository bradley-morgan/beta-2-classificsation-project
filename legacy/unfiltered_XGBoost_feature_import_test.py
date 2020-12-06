from tools.DatasetCompiler import DatasetCompiler
from sklearn.metrics import matthews_corrcoef, make_scorer
import numpy as np
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


data = DatasetCompiler.load_from_pickle('../data/processed/non-filtered/lrg_clean_data_v2.pickle')
vals, counts_after_sampling = np.unique( data.y_train, return_counts=True)

# Fit a model on whole training dataset using either optimised or default params for testing
# TODO This should be the optimised pretrained model but im just using a default for testing
# optimised_model = XGBClassifier(
#     n_estimators=979,
#     max_depth=19,
#     learning_rate=0.07944,
#     subsample=0.851,
#     colsample_bytree=0.7626,
#     booster='gbtree',
#     gamma=0.2376,
#     eta=0.4664,
#     min_child_weight=0.08316,
#     max_delta_step=10,
#     reg_alpha=0.01635,
#     reg_lambda=0.6584,
#     scale_pos_weight=4,
#     tree_method='gpu_hist'
#
# )
optimised_model = XGBClassifier(tree_method='gpu_hist')
optimised_model.fit(data.x_train, data.y_train)
optimised_model_ypreds = optimised_model.predict(data.x_hold_out)
optimised_model_hold_out_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=optimised_model_ypreds)#

# Compute Permutation Importance
scorer = make_scorer(matthews_corrcoef)
train_feature_importance = permutation_importance(optimised_model,
                                                  data.x_train,
                                                  data.y_train,
                                                  scoring=scorer,
                                                  n_repeats=3,
                                                  n_jobs=1)

feature_importances = pd.DataFrame(zip(data.feature_names, train_feature_importance["importances_mean"]), columns=['Feature Names','Permutation Score MCC'])
feature_importances.sort_values(by='Permutation Score MCC', ascending=False, inplace=True)

# Threshold Method For Feature Selection
n_features_to_test = 100
feature_import_scores = []
number_of_features = []
with tqdm(total=n_features_to_test, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
          desc=f'Calculating Optimal Features for Model Training') as progress_bar:
    for idx, (_, row) in enumerate(feature_importances.iterrows()):

        if idx > n_features_to_test:
            break

        feature_name = row['Feature Names']
        importance_score = row['Permutation Score MCC']
        # Select features based on current threshold

        selected_features = feature_importances[feature_importances['Permutation Score MCC'] >= importance_score]
        selected_idx = selected_features.index.values
        selected_x_train = data.x_train[:, selected_idx]
        selected_x_hold_out = data.x_hold_out[:, selected_idx]

        # Train model on selected features
        selection_model = XGBClassifier()
        selection_model.fit(selected_x_train, data.y_train)
        #Eval the newly trained model on hold out
        selected_model_y_preds = selection_model.predict(selected_x_hold_out)
        selected_model_hold_out_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=selected_model_y_preds)
        feature_import_scores.append(selected_model_hold_out_score)
        number_of_features.append(selected_x_train.shape[1])
        progress_bar.update(1)


sns.set()
fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
ax.plot(feature_import_scores, marker='o')
ax.plot([optimised_model_hold_out_score]*len(feature_import_scores), color='r')
ax.legend(['Feature Selection Model', 'Model Trained on All Featured'])
ax.set(xlabel='Thresholds', ylabel='Model Hold Out Performance(MCC)')
ax.set_xticks(range(0, len(number_of_features)))
# Set ticks labels for x-axis
ax.set_xticklabels(number_of_features)
ax.grid(b=True)
plt.title('Model Performance as a Function of Features')
plt.show()

#
# plt.bar(range(len(optimised_model.feature_importances_)), optimised_model.feature_importances_)
# plt.show()
#