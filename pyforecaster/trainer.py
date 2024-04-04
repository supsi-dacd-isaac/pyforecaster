from sklearn.model_selection import cross_validate as sklear_cv
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import optuna
import pandas as pd
from pyforecaster.metrics import nmae, make_scorer
from functools import partial
import numpy as np
from pyforecaster.dictionaries import HYPERPAR_MAP
from inspect import signature

def default_param_space_fun(trial):
    param_space = {'model__n_estimators': trial.suggest_int('n_estimators', 10, 300),
                   'model__learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2)}
    return param_space

def get_score_key(cv_results, scoring, score_key=None):
    fallback_name = 'test_' + list(scoring.keys())[0] if isinstance(scoring, dict) else list(cv_results.keys())[-1]
    fallback_name = 'test_score' if 'test_score' in cv_results.keys() else fallback_name
    score_key = score_key if score_key is not None else fallback_name
    return score_key

def cross_validate(x: pd.DataFrame, y: pd.DataFrame, model, cv_folds, scoring=None, cv_type='full',
                   trial=None, storage_fun=None, score_key=None, **cv_kwargs):
    """
    :param x: pd.DataFrame of features
    :param y: pd.DataFrame of target
    :param model: learning model
    :param cv_folds: list of tuples with train and test indexes
    :param scoring: scoring function, see sklearn.metrics.make_scorer
    :param cv_type: type of cross validation. If full, standard CV is performed, if random_fold, a random fold is
                    selected at each iteration
    :param trial:  optuna trial, optional
    :param storage_fun: function to store some results of the model. It must take as input the storage and the results
    :param cv_kwargs: additional arguments to be passed to the cross validation function
    :return:
    """
    # create generator
    cv_gen = (f for f in cv_folds)

    # retrieve cross validation score
    if cv_type == 'full':
        if storage_fun is not None:
            cv_kwargs.update({'return_estimator': True})
        cv_results = sklear_cv(model, x, y, scoring=scoring, cv=cv_gen, **cv_kwargs)
        score_key = get_score_key(cv_results, scoring, score_key=score_key)
        scores = cv_results[score_key]
        score = np.mean(scores)

        if trial is not None:
            trial.set_user_attr('cv_test_scores', scores)

        if storage_fun:
            cv_results['y_te'] = {'fold_{}'.format(i): y.loc[cv_idxs[1]] for i, cv_idxs in enumerate(cv_folds)}
            cv_results['y_hat'] = {'fold_{}'.format(i): m.predict(x.loc[cv_idxs[1]]) for i, cv_idxs, m in
                                   zip(range(len(cv_folds)), cv_folds, cv_results['estimator'])}
    elif cv_type == 'random_fold':
        tr_idx, te_idx = cv_folds[np.random.choice(len(cv_folds), 1)[0]]
        x_tr, x_te, y_tr, y_te = x.loc[tr_idx], x.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
        model.fit(x_tr, y_tr)
        y_hat = model.predict(x_te)
        cv_results = {k:np.atleast_1d(s(model, x_te, y_te)) for k, s in scoring.items()} if isinstance(scoring, dict) else {'test_score': np.atleast_1d(scoring(model, x_te, y_te))}
        score_key = get_score_key(cv_results, scoring, score_key=score_key)
        score = cv_results[score_key]

        if storage_fun:
            cv_results['y_te'] = y_te
            cv_results['y_hat'] = y_hat
    else:
        raise ValueError('cv_type must be either "full" or "random_fold"')

    return cv_results, score


def optuna_objective(trial, x: pd.DataFrame, y: pd.DataFrame, model, cv, scoring=None, param_space_fun=default_param_space_fun,
              cv_type='full', storage=None, storage_fun=None, **cv_kwargs):
    """
    :param trial: optuna trial
    :param x: pd.DataFrame of features
    :param y: pd.DataFrame of target
    :param model: learning model
    :param cv: cross validation generator
    :param scoring: scoring function, see sklearn.metrics.make_scorer
    :param param_space_fun: function to set the parameter space of the model
    :param cv_type: type of cross validation. If full, standard CV is performed, if random_fold, a random fold is
                    selected at each iteration
    :param storage: list to store some results of the model
    :param storage_fun: function to store some results of the model. It must take as input the storage and the results
    :param cv_kwargs: additional arguments to be passed to the cross validation function
    :return:
    """

    # set optuna parameter space
    params = param_space_fun(trial)
    if not np.all([k in list(signature(model.__class__).parameters.keys()) + list(model.get_params().keys()) for k in params.keys()]):
        raise ValueError('not all parameter space names are in the model signature. This means they will not affect the'
                         ' performance of your model. Something is wrong')
    model.set_params(**params)

    # retrieve cross validation score
    cv_results, score = cross_validate(x, y, model, cv, scoring=scoring, cv_type=cv_type, trial=trial,
                                       storage_fun=storage_fun, **cv_kwargs)

    # this is thought to permanently store some function outputs that take as input the current trained model
    if storage_fun is not None:
        storage_fun(storage, cv_results)

    return score


def hyperpar_optimizer(x, y, model, n_trials=40, metric=None, cv=5, param_space_fun=None,
                       cv_type='full', sampler=None, callbacks=None, storage_fun=None, formatter=None, **cv_kwargs):
    """
    Function to optimize hyperparameters of a model using optuna
    :param x: pd.DataFrame of features
    :param y: pd.DataFrame of target
    :param model: learning model
    :param cv: cross validation generator
    :param scoring: scoring function, see sklearn.metrics.make_scorer
    :param param_space_fun: function to set the parameter space of the model
    :param cv_type: type of cross validation. If full, standard CV is performed, if random_fold, a random fold is
                    selected at each iteration
    :param storage: list to store some results of the model
    :param storage_fun: function to store some results of the model. It must take as input the storage and the results
    :param cv_kwargs: additional arguments to be passed to the cross validation function
    :return:
    """
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
    if isinstance(cv, int) or isinstance(cv, float):
        if formatter is not None:
            fold_generator = formatter.time_kfcv(x.index, int(cv))
            cv = list(fold_generator)
        else:
            cv = list(KFold(int(cv)).split(x, y))
    elif cv is not list:
        cv = list(cv)


    if storage_fun is not None:
        stored_replies = []
    else:
        stored_replies = None
    obj_fun = partial(optuna_objective, x=x, y=y, model=model, cv=cv, scoring=scoring, param_space_fun=param_space_fun,
                      cv_type=cv_type, storage_fun=storage_fun, storage=stored_replies, **cv_kwargs)

    study.optimize(obj_fun, n_trials=n_trials, callbacks=callbacks)

    return study, stored_replies


def base_storage_fun(storage, cv_results):
    reply = cv_results['y_hat']
    storage.append(reply)


def retrieve_cv_results(study):
    trials_df = study.trials_dataframe()
    usr_attr_cols = [c for c in trials_df.columns if 'user_attr' in c]
    if len(usr_attr_cols) > 0:
        trials_df['cv_test_score_std'] = np.vstack(trials_df[usr_attr_cols[0]]).std(axis=1)

    trials_df['rank'] = trials_df['value'].rank().astype(int)
    return trials_df