from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import tools.model_tools as m_tools
import tools.cloud_tools as c_tools

def make_model(m_config):

    if m_config.load_model_from == 'cloud':
        model = c_tools.cloud_load(m_config.model_file_name, m_config.global_load_run_path)
        return model

    elif m_config.load_model_from == 'local':
        model = m_tools.local_load_model(m_config.model_file_name)
        return model

    elif m_config.load_model_from == 'train':
        return load_training_model(m_config)

    else:
        raise ValueError(f'{m_config.load_model_from} not a valid option, select cloud, local or train')


def load_training_model(m_config):
    if m_config.model_type == 'decision_tree' or m_config.model_type == 'd_tree' or m_config.model_type == 'DecisionTree' or m_config.model_type == 'FIMP-dt':
        return make_decision_tree
    else:
        raise ValueError(
            f'{m_config.model_type} does not exist as possible model to create. Please implement in make_models')


def make_decision_tree(m_config):
    if m_config.test_mode:
        return DecisionTreeClassifier(random_state=1)

    else:
        return DecisionTreeClassifier(
            criterion=m_config.criterion,
            splitter=m_config.splitter,
            max_depth=m_config.max_depth,
            max_features=m_config.max_features,
            min_samples_split=m_config.min_samples_split,
            min_samples_leaf=m_config.min_samples_leaf,
            class_weight=m_config.class_weight,
            random_state=1
        )


def make_xgboost(m_config):
    if m_config.test_mode:
        return XGBClassifier()

    else:
        return XGBClassifier()







