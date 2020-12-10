
from tools.feature_importance_estimation import FeatureImportances
from tools.make_models import make_model
import Config_Script
import os
print(f'Program Running: {os.path.basename(__file__)}')

meta_data = Config_Script.get_config().feature_importance_config

feat_import = FeatureImportances(meta_data, make_model(meta_data.model), meta_data.target_datasets, meta_data.cloud_log)
feat_import.get_feature_importances()