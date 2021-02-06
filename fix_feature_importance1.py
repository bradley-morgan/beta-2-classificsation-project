from tools.DatasetCompiler import DatasetCompiler
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tools.cloud_tools as c_tools
import wandb
from tools.ImageSaver import ImageSaver
import seaborn as sns
import matplotlib.pyplot as plt

src = 'data/processed/non-filtered/dataset1-2percent-hold-out.pickle'
model_name = 'v4_CV-XGBoost.joblib'
run_path = 'bradamorg/B2AR-Unfiltered-Hotfix/1ai6hzpc'
data = DatasetCompiler.load_from_local(src).to_dict()

run = wandb.init(project='B2AR-Unfiltered-Hotfix', name='Feature Importance hold out')
image_saver = ImageSaver(run)
x_train = data['x_train']
y_train = data['y_train']

model = XGBClassifier(num_class=2)
# model.fit(x_train, y_train)

shap_vals = shap.TreeExplainer(model).shap_values(data['x_hold_out'])
neg_shaps = shap_vals[0]
pos_shaps = shap_vals[1]

sns.set()
shap.summary_plot(
    neg_shaps,
    data['x_hold_out'],
    feature_names=data['feature_names'],
    show=False,
    plot_type='dot',
    max_display=10
)
plt.tight_layout()
image_saver.save(plot=plt.gcf(),
                      name=f'Local Feature Importance Negative Class', format='png')
plt.clf()

sns.set()
shap.summary_plot(
    pos_shaps,
    data['x_hold_out'],
    feature_names=data['feature_names'],
    show=False,
    plot_type='dot',
    max_display=10
)
plt.tight_layout()
image_saver.save(plot=plt.gcf(),
                      name=f'Local Feature Importance Positive Class', format='png')
plt.clf()

a = 0
#' data=shap.sample(x_train), model_output='probability'