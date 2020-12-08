import wandb
from joblib import dump, load
import os
import h5py

def sweep_init(meta_data, is_sweep):

    if is_sweep:
        return wandb.init(
            config=meta_data,
            project=meta_data.project_name,
            notes=meta_data.notes,
            allow_val_change=True,
        )

    else:
        return wandb.init(
        config=meta_data,
        project=meta_data.project_name,
        notes=meta_data.notes,
        allow_val_change=True,
        name=meta_data.run_name,
    )


def cloud_save(data, file_name, run):
    if not file_name.endswith('.joblib'):
        file_name = f'{file_name}.joblib'

    dump(data, os.path.join(run.dir, file_name))
    print(f'Cloud save {file_name}: Model Successfully Saved')


def cloud_load(file_name, run_path):
    output_buffer = wandb.restore(file_name, run_path=run_path).name
    output = load(output_buffer)
    return output
