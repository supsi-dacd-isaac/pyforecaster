from sklearn.model_selection import cross_validate
import optuna
import pandas as pd
from pyforecaster.metrics import nmae
from functools import partial
import numpy as np


def default_param_space_fun(trial):
    param_space = {'model__n_estimators': trial.suggest_int('n_estimators', 10, 300),
                   'model__learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.2)}
    return param_space


def objective(trial, x: pd.DataFrame, y: pd.DataFrame, model, cv, scoring=nmae, param_space_fun=default_param_space_fun,
              hpo_type='full', **cv_kwargs):
    """
    :param trial: optuna trial
    :param x: feature set
    :param y: y training set
    :param model: a single model or a sklearn Pipeline. If Pipeline, param_space dict must be in the form
    {'model__c':trial.suggest_uniform('name', 0, 3)}
    :param param_space_fun: accept an optuna.Trial instance and return a dict of parameter spaces
    :param cv: number or generator
    :param hpo_type: type of hyper parameter optimization
    :param cv_kwargs:
    :param scoring: str, callable, list, tuple, or dict
    :return:
    """

    # set optuna parameter space
    param_space = param_space_fun(trial)
    model.set_params(**param_space)
    # create generator
    cv_gen = (f for f in cv)

    # retrieve cross validation score
    if hpo_type == 'full':
        cv_results = cross_validate(model, x, y, scoring=scoring, cv=cv_gen, **cv_kwargs)
        scores = cv_results['test_score']
        trial.set_user_attr('cv_test_scores', scores)
        score = np.mean(scores)
    elif hpo_type == 'one_fold':
        tr_idx, te_idx = list(cv_gen)[0]
        x_tr, x_te, y_tr, y_te = x.loc[tr_idx], x.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
        model.fit(x_tr, y_tr)
        preds = model.predict(x_te)
        score = scoring(preds, y_te)
    elif hpo_type == 'random_fold':
        tr_idx, te_idx = np.random.choice(list(cv), 1)
        x_tr, x_te, y_tr, y_te = x.loc[tr_idx], x.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
        model.fit(x_tr, y_tr)
        preds = model.predict(x_te)
        score = scoring(preds, y_te)
    else:
        raise ValueError

    return score


def hyperpar_optimizer(x, y, model, n_trials, scoring, cv, param_space_fun=default_param_space_fun,
                       hpo_type='full', sampler=None, **cv_kwargs):
    if sampler is None:
        sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if cv is not list:
        cv = list(cv)
    obj_fun = partial(objective, x=x, y=y, model=model, cv=cv, scoring=scoring, param_space_fun=param_space_fun,
                      hpo_type=hpo_type, **cv_kwargs)
    study.optimize(obj_fun, n_trials=n_trials)
    return study


def retrieve_cv_results(study):
    trials_df = study.trials_dataframe()
    usr_attr_cols = [c for c in trials_df.columns if 'user_attr' in c]
    if len(usr_attr_cols) > 0:
        trials_df['cv_test_score_std'] = np.vstack(trials_df[usr_attr_cols[0]]).std(axis=1)

    trials_df['rank'] = trials_df['value'].rank().astype(int)
    return trials_df