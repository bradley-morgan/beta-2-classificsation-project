
class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasets: dict):

        value = self.config['value']

        keys = list(datasets.keys())

        for name in keys:
            datasets[name]['data'].fillna(value, inplace=True)
            nan_count = datasets[name]['data'].isna().sum().sum()
            if nan_count > 0:
                raise ValueError(f'Transform Error: Change nans has failed. transform detected {nan_count} NaNs Remaining')

        return datasets