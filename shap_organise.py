import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

shaps = joblib.load('analysis/SHAPS/rfc_shaps_x_hold_out.joblib')
x_data = joblib.load('analysis/SHAPS/rfc_data_x_hold_out.joblib')
y_data = joblib.load('analysis/SHAPS/rfc_data_y_hold_out.joblib')
features = joblib.load('analysis/SHAPS/feature_names.joblib')

plot = shap.summary_plot(
    shap_values=shaps,
    features=x_data,
    feature_names=features,
    show=False,
    plot_type='dot',
    max_display=10
)
plt.tight_layout()
plt.show()


# shaps = pd.DataFrame(shaps, columns=features)
# x_data = pd.DataFrame(x_data, columns=features)
#
# shaps.columns = shaps.columns[np.argsort(np.abs(shaps).mean(0))[::-1][:len(features)]]
# x_data.columns = x_data.columns[np.argsort(np.abs(shaps).mean(0))[::-1][:len(features)]]
# x_data['Action'] = y_data
#
# shaps.to_csv('./analysis/SHAPS/Output/rfc/hold_out/shaps_hold_out.csv', index=False)
# x_data.to_csv('./analysis/SHAPS/Output/rfc/hold_out/data_hold_out.csv', index=False)

a = 0