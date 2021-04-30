from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

# Define pipelines
nb_pipe = Pipeline([("vect", CountVectorizer(analyzer="char_wb")),
                    ("feat", SelectKBest(chi2)), 
                    ("clf", MultinomialNB())
                    ])

svm_pipe = Pipeline([("vect", CountVectorizer(analyzer="char_wb")),
                    ("feat", SelectKBest(chi2)), 
                    ("clf", SGDClassifier())
                    ])

xgb_pipe = Pipeline([("vect", CountVectorizer(analyzer="char_wb")),
                    ("feat", SelectKBest(chi2)), 
                    ("clf", XGBClassifier(use_label_encoder=False))
                    ])

# Grid search params
ngram = [(1, 3), (1, 4), (1, 5)]
k = np.array(range(800, 1601, 200))

grid_nb = [{
    "vect__ngram_range" : ngram,
    "feat__k" : k,
    "clf__alpha" : (0.5, 0.75, 1)
    }]

grid_svm = [{
    "vect__ngram_range" : ngram,
    "feat__k" : k,
    "clf__alpha" : [1e-4, 1e-3, 1e-2, 1e-1],
    "clf__max_iter" : [10, 20, 50, 100, 200, 300, 1000]
    }]

grid_xgb = [{
    "vect__ngram_range" : ngram,
    "feat__k" : k,
    "clf__max_depth" : [3, 4, 5, 6, 7, 8, 9, 10],
    "clf__min_child_weight" : [0.1, 1, 100,1000],
    "clf__gamma" : np.arange(0.01, 30, 0.01),
    "clf__subsample": np.arange(0.01, 1.0, 0.01)
    }]