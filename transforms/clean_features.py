

class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasets: dict):

        keys = datasets.keys()

        for key in keys:

            for col_name in datasets[key]['data'].columns:

                if col_name in self.config['exceptions']:
                    continue

                new_col_names = col_name
                isLetterA = new_col_names.startswith("A")

                if isLetterA:
                    new_col_names = new_col_names[3:]


                else:
                    new_col_names = new_col_names[2:]

                if new_col_names.startswith("0"):
                    new_col_names = new_col_names[1:]

                datasets[key]['data'].rename({col_name: new_col_names}, axis=1, inplace=True)

        return datasets