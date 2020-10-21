from src.utils.set_path import path
from sklearn import tree
import wandb
import os
import subprocess

class ImageSaver:

    def __init__(self):
        self.save_dir = path('../tmp')

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.clean_up()

    def save(self, plot, run: wandb.run, name: str, format: str):

        save_path = os.path.join(self.save_dir, f'{name}.{format}')
        #TODO This needs Testing
        plot.savefig(save_path, format=format, dpi=300)

        run.log({name: wandb.Image(save_path)})

    def save_graphviz(self, model: tree.DecisionTreeClassifier,
                      run: wandb.run,
                      feature_names: list,
                      class_names: list,
                      graph_name: str,):

        name = 'tree_graph'
        format = 'dot'

        dot_out_file = os.path.join(self.save_dir, f'{name}.{format}')
        tree.export_graphviz(
            model,
            out_file=dot_out_file,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
        )
        # Convert to png
        format = 'png'
        png_out_file = os.path.join(self.save_dir, f'{name}.{format}')
        out = subprocess.run(['dot', '-Tpng', dot_out_file, '-o', png_out_file])

        run.log({graph_name: wandb.Image(png_out_file)})

        if out.returncode != 0:
            raise ValueError('ImageSave.save_graphviz: Graphviz dot to png command failed during subprocess run')

    def clean_up(self):
        # Clear tmp folder of files no longer needed
        files = os.listdir(self.save_dir)
        for file in files:
            os.remove(os.path.join(self.save_dir, file))




