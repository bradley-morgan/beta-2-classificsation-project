import unittest
from src.processing.DatasetCompiler import DatasetCompiler
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.drop_nans import DropNans
from transforms.clean_feature_names import CleanFeatureNames


class TestFeatures(unittest.TestCase):

    def test_feature_names(self):
        datasetConfig = {
            'src': '../data/raw',
            'name': 'dataset test',
            'log_data': False,
            'labels': 'target',
            'notes': 'Data contains B2in, Z, R and Merged datasets',
            'test_size': 0.1,
            'stats': dict(
                names=[],
            ),
            'transforms': {
                'merge_datasets': dict(
                    merge_all=True,
                    merge_all_name='merge',
                    groups=[('B2in-ant', 'B2in-ag'), ('R-ant', 'R-ag'), ('Z-ant', 'Z-ag')],
                    group_names=['B2in', 'R', 'Z']
                ),
                'change_nans': dict(
                    value=0
                ),
                'drop_nans': dict(
                    target_datasets=['B2in', 'R', 'Z']
                )
            }
        }

        dataset = DatasetCompiler(config=datasetConfig, project='unittest')
        merge_datasets = MergeDatasets(config=datasetConfig["transforms"]["merge_datasets"])
        clean_feature_names = CleanFeatureNames()

        dataset.load()
        dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
        dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
        dataset.remove_feature(feature_name='Ligand_Pose')

        dataset = clean_feature_names(datasetComplier=dataset)
        dataset = merge_datasets(datasetComplier=dataset)

        test_sample = "113/OD2 - /1/N"
        dataB2in = dataset.datasets["B2in"]["data"]
        dataR = dataset.datasets["R"]["data"]
        dataZ = dataset.datasets["Z"]["data"]

        assertB2in = any([True if col.startswith(test_sample) else False for col in dataB2in.columns])
        assertR = any([True if col.startswith(test_sample) else False for col in dataR.columns])
        assertZ = any([True if col.startswith(test_sample) else False for col in dataZ.columns])

        self.assertEqual(any([assertB2in, assertR, assertZ]), True, f"Could not find test sample {test_sample} in all datasets")

if __name__ == '__main__':
    unittest.main()