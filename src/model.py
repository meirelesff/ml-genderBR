from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold
from config import settings
import pipelines as pps
import pandas as pd
import numpy as np
import joblib
import os


# Function to prepare names sample
def prepare_sample(arq, settings):

    # Load data
    nomes = pd.read_csv(arq)
    nomes["first_name"] = nomes.first_name.str.lower().str.strip()

    # Sample rows
    if settings.sample_size == "all":
        pass
    else:
        nomes = nomes.sample(settings.sample_size, random_state=settings.seed)

    # Sample gender
    np.random.RandomState(seed=settings.seed)
    y = np.random.binomial(1, nomes.prop_female)
    x = "_" + nomes.first_name + "_"
    
    return y, x


# Function to train models
def train_models(y, x, model, pipe, grid, settings, random_cv=False):

    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("results"):
        os.makedirs("results")

    # Training
    rcv = RepeatedKFold(n_splits=settings.splits, n_repeats=settings.repeats,
                        random_state=settings.seed)

    if not random_cv:
        gs = GridSearchCV(pipe, grid, cv=rcv, n_jobs=settings.threads, 
                      scoring="accuracy").fit(x, y)
    else:
        gs = RandomizedSearchCV(pipe, grid, cv=rcv, n_jobs=settings.threads, 
                      scoring="accuracy", n_iter=settings.n_iter,
                      random_state=settings.seed).fit(x, y, clf__eval_metric="logloss")

    # Save training logs and best model
    pd.DataFrame(gs.cv_results_).to_csv("results/" + model + "_train.csv", index=False)
    joblib.dump(gs.best_estimator_, "src/models/best_" + model + ".pkl")



if __name__ == "__main__":
    y, x = prepare_sample("raw_data/nomes.csv", settings)
    train_models(y, x, "nb", pps.nb_pipe, pps.grid_nb, settings)
    train_models(y, x, "svm", pps.svm_pipe, pps.grid_svm, settings)
    train_models(y, x, "xgb", pps.xgb_pipe, pps.grid_xgb, settings, random_cv=True)
