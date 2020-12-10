from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import tools.model_tools as m_tools
import tools.cloud_tools as c_tools


def make_model(model_type):
    if model_type == 'decision_tree' or model_type == 'd_tree' or model_type == 'DecisionTree' or model_type == 'FIMP-dt':
        return decision_tree_contrustor

    elif model_type == 'xgboost' or model_type == 'xgb' or model_type == 'XGBoost' or model_type == 'FIMP-xg':
        return xgboost_constructor
    else:
        raise ValueError(
            f'{model_type} does not exist as possible model to create. Please implement in make_models')


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
        return XGBClassifier(tree_method='gpu_hist')
    else:
        return XGBClassifier(
            n_estimators=m_config.n_estimators,
            max_depth=m_config.max_depth,
            learning_rate=m_config.learning_rate,
            subsample=m_config.subsample,
            colsample_bytree=m_config.colsample_bytree,
            booster=m_config.booster,
            gamma=m_config.gamma,
            eta=m_config.eta,
            min_child_weight=m_config.min_child_weight,
            max_delta_step=m_config.max_delta_step,
            reg_alpha=m_config.reg_alpha,
            reg_lambda=m_config.reg_lambda,
            scale_pos_weight=m_config.scale_pos_weight,
            tree_method='gpu_hist'
        )


def decision_tree_contrustor(m_config):
    if m_config.load_model_from == 'cloud':
        model = c_tools.cloud_load(m_config.model_file_name, m_config.global_load_run_path)
        return model

    elif m_config.load_model_from == 'local':
        model = m_tools.local_load_model(m_config.model_file_name)
        return model.model

    elif m_config.load_model_from == 'train':
        return make_decision_tree(m_config)

    else:
        raise ValueError(f'{m_config.load_model_from} not a valid option, select cloud, local or train')


def xgboost_constructor(m_config):
    if m_config.load_model_from == 'cloud':
        model = c_tools.cloud_load(m_config.model_file_name, m_config.global_load_run_path)
        return model

    elif m_config.load_model_from == 'local':
        model = m_tools.local_load_model(m_config.model_file_name)
        return model.model

    elif m_config.load_model_from == 'train':
        return make_xgboost(m_config)

    else:
        raise ValueError(f'{m_config.load_model_from} not a valid option, select cloud, local or train')
