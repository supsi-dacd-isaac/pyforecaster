from sklearn.model_selection import cross_validate, KFold
import optuna
import pandas as pd
from pyforecaster.metrics import nmae, make_scorer
from functools import partial
import numpy as np
from pyforecaster.dictionaries import HYPERPAR_MAP


def default_param_space_fun(trial):
    param_space = {'model__n_estimators': trial.suggest_int('n_estimators', 10, 300),
                   'model__learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.2)}
    return param_space


def objective(trial, x: pd.DataFrame, y: pd.DataFrame, model, cv, scoring=None, param_space_fun=default_param_space_fun,
              hpo_type='full', storage=None, storage_fun=None, **cv_kwargs):
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
        if storage_fun is not None:
            cv_kwargs.update({'return_estimator': True})
        cv_results = cross_validate(model, x, y, scoring=scoring, cv=cv_gen, **cv_kwargs)
        scores = cv_results['test_score']
        trial.set_user_attr('cv_test_scores', scores)
        score = np.mean(scores)
        if storage_fun:
            cv_results['y_te'] = {'fold_{}'.format(i): y.loc[cv_idxs[1]] for i, cv_idxs in enumerate(cv)}
            cv_results['y_hat'] = {'fold_{}'.format(i): m.predict(x.loc[cv_idxs[1]]) for i, cv_idxs, m in
                                   zip(range(len(cv)), cv, cv_results['estimator'])}
    elif hpo_type == 'one_fold':
        tr_idx, te_idx = list(cv_gen)[0]
        x_tr, x_te, y_tr, y_te = x.loc[tr_idx], x.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
        model.fit(x_tr, y_tr)
        y_hat = model.predict(x_te)
        score = scoring(model, x_te, y_te)
        cv_results = {'estimator': model, 'y_te': y_te, 'y_hat': y_hat}
    elif hpo_type == 'random_fold':
        tr_idx, te_idx = np.random.choice(list(cv), 1)
        x_tr, x_te, y_tr, y_te = x.loc[tr_idx], x.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
        model.fit(x_tr, y_tr)
        y_hat = model.predict(x_te)
        score = scoring(model, x_te, y_te)
        cv_results = {'estimator': model, 'y_te': y_te, 'y_hat': y_hat}
    else:
        raise ValueError

    # this is thought to permanently store some function outputs that take as input the current trained model
    if storage_fun is not None:
        storage_fun(storage, cv_results)

    return score


def hyperpar_optimizer(x, y, model, n_trials=40, metric=None, cv=5, param_space_fun=None,
                       hpo_type='full', sampler=None, callbacks=None, storage_fun=None,  **cv_kwargs):
    # set default scorer
    scoring = make_scorer(metric) if metric is not None else make_scorer(nmae)

    # set default param_space_function if needed
    forecaster_name = model.__class__.__name__
    def_param_space = HYPERPAR_MAP[forecaster_name] if forecaster_name in HYPERPAR_MAP.keys()\
        else default_param_space_fun
    param_space_fun = def_param_space if param_space_fun is None else param_space_fun

    if sampler is None:
        sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if cv is not list:
        if isinstance(cv, int) or isinstance(cv, float):
            cv = list(KFold(int(cv)).split(x, y))
        cv = list(cv)

    if storage_fun is not None:
        stored_replies = []
    else:
        stored_replies = None
    obj_fun = partial(objective, x=x, y=y, model=model, cv=cv, scoring=scoring, param_space_fun=param_space_fun,
                      hpo_type=hpo_type, storage_fun=storage_fun, storage=stored_replies, **cv_kwargs)

    study.optimize(obj_fun, n_trials=n_trials, callbacks=callbacks)

    return study, stored_replies


def base_storage_fun(storage, results):
    reply = results['y_hat']
    storage.append(reply)


def retrieve_cv_results(study):
    trials_df = study.trials_dataframe()
    usr_attr_cols = [c for c in trials_df.columns if 'user_attr' in c]
    if len(usr_attr_cols) > 0:
        trials_df['cv_test_score_std'] = np.vstack(trials_df[usr_attr_cols[0]]).std(axis=1)

    trials_df['rank'] = trials_df['value'].rank().astype(int)
    return trials_df