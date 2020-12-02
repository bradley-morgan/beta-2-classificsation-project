from tools.DatasetCompiler import DatasetCompiler
import numpy as np

data = DatasetCompiler.load_from_pickle('./data/processed/clean_data.pickle')

vals, counts = np.unique(data.y_train, return_counts=True)

min_percentage = counts[0] / sum(counts) * 100
a = 0