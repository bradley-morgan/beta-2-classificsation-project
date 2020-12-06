from tools.model_performance_estimation import ModelEstimations
from tools.make_models import make_model
import Filter_Script
import os
print(f'Program Running: {os.path.basename(__file__)}')


meta_data = Filter_Script.get_config().model_estimation_config

emp = ModelEstimations(meta_data, make_model(meta_data.model), cloud_log=meta_data.cloud_log)
# emp.estimate_population_variance(meta_data.test_repeats, meta_data.n_samples)
# emp.estimate_model_uncertainty(
#     meta_data.confidence_level, meta_data.n_repeats, meta_data.n_samples
# )
emp.estimate_n_repeats(0.95,  0.08271353146450434, (0.01, 0.001), 15)

