from src.utils.set_path import path
import wandb
import os

class ImageSaver:

    def __init__(self):
        self.save_dir = path('../tmp')

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.clean_up()

    def save(self, plot, run: wandb.run, name: str, format: str, plot_type='seaborn'):

        save_path = os.path.join(self.save_dir, f'{name}.{format}')
        # if plot_type == 'seaborn' or plot_type == 'sns':
        #     plot = plot.get_figure()
        #     plot.savefig(save_path, format=format, dpi=300)

        # elif plot_type == 'matplotlib' or plot_type == 'mat':
        #TODO This needs Testing
        plot.savefig(save_path, format=format, dpi=300)

        run.log({name: wandb.Image(save_path)})

    def clean_up(self):
        # Clear tmp folder of files no longer needed
        files = os.listdir(self.save_dir)
        for file in files:
            os.remove(os.path.join(self.save_dir, file))




