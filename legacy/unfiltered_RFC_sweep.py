import wandb
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from configs import get
from tools.DatasetCompiler import DatasetCompiler

# Setup
parameters = get.general_config(return_obj=False)
parameters['src'] = './data/processed/lrg_clean_data.pickle'
parameters['project_name'] = 'test'

wandb.init(
           config=parameters,
           project=parameters['project_name'],
           notes=parameters['notes'],
           allow_val_change=True
          )
config = wandb.config

if config.class_weights == "None":
    wandb.config.update({"class_weights": None}, allow_val_change=True)
    print('CONFIG UPDATE')

# Load Data_set
data = DatasetCompiler.load_from_local(config.src)

cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

model = RandomForestClassifier(

)

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=data.x_train, y=data.y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)

mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
wandb.log(metrics)





