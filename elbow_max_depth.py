from tools.model_performance_estimation import CrossValidation
from sklearn.tree import DecisionTreeClassifier
from tools.DatasetCompiler import DatasetCompiler
import wandb
from tools.ImageSaver import ImageSaver
from tools.make_models import make_model
from tools import model_tools as m_tools
import Config_Script
import matplotlib.pyplot as plt
import seaborn as sns
import tools.cloud_tools as c_tools


config = Config_Script.get_config()
config = config.cross_validation_config.config
config.run_name = 'Elbow Sweep Max Depth'

run = c_tools.sweep_init(config, is_sweep=False)
config = wandb.config
image_saver = ImageSaver(run)

data = DatasetCompiler.load_from_local(config.src)

# ================================== Run Cross Validation =======================================
depths = 20
mean_performances = []
for i in range(depths, 0, -1):
    config.update({'max_depth': i}, allow_val_change=True)
    results = CrossValidation(
        k_folds=10, n_repeats=3, data=data,
        make_model_func=make_model(config.model), model_parameters=config
    ).run()
    desc_stats = m_tools.get_descriptive_stats(results.cross_val_mcc_scores)
    mean_performances.append(desc_stats.mean)

sns.set()
fig1, ax2 = plt.subplots(1, 1, figsize=(18,8), dpi= 100)
ax2.plot(mean_performances, marker='.')
ax2.set(xlabel='Max Depth', ylabel='Mean MCC')
ax2.set_xticks(range(0, len(range(depths, 0, -1))))
ax2.set_xticklabels(range(depths, 0, -1))
ax2.grid(b=True)
plt.title('Best Estimators')

image_saver.save(plt.gcf(), name='Eblow: Max Depth', format='png')