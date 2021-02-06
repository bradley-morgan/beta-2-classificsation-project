
class Transform:

    def __init__(self, config):
        self.config = config

    def __call__(self, datasets: dict):

        merge_all = self.config['merge_all']
        merge_all_name = self.config['merge_all_name']
        groups = self.config['groups']
        group_names = self.config['group_names']

        out_dataset = {}
        if groups is not None:
            for group, group_name in zip(groups, group_names):
                group = list(group)
                f_key = group.pop()
                merged_df = datasets[f_key]['data']

                for df_name in group:
                    subset = datasets[df_name]['data']
                    merged_df = merged_df.append(subset, sort=False)

                out_dataset[group_name] = {'data': merged_df}
        else:
            out_dataset = datasets
            group_names = list(out_dataset.keys())

        if merge_all:
            j_key = group_names.pop()
            merged_df = out_dataset[j_key]['data']
            for name in group_names:
                subset = out_dataset[name]['data']
                merged_df = merged_df.append(subset, sort=False,)

            out_dataset[merge_all_name] = {'data': merged_df}

        return out_dataset
