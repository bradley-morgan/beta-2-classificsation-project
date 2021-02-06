from tools.Data import Preprocess


train_processor = Preprocess(config_src='./configs/train_preprocessing_unfiltered.yaml')
train_processor.transform()

zinc_processor = Preprocess(config_src='./configs/ZINC_preprocessing_unfiltered.yaml')
zinc_processor.transform()

train_features = set(train_processor.datasets['merged-data']['data'].columns.values)
zinc_features = set(zinc_processor.datasets['merged-data']['data'].columns.values)

shared_features = train_features.intersection(zinc_features)
not_shared = train_features.symmetric_difference(zinc_features)
a = 0