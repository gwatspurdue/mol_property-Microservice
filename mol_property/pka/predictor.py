# -*- coding: utf-8

import os
import joblib
import xgboost as xgb
import numpy as np
from .data_utils import DataUtils

class PkaPredictor(object):

    def __init__(self, model_dir=os.path.join(os.path.dirname(__file__), "model"), feature_type="morgan+macc"):
        self._load_model(clf_modelpath=os.path.join(model_dir, "pka_classification.pkl"),
                         acidic_modelpath=os.path.join(model_dir, "pka_acidic_regression.pkl"),
                         basic_modelpath=os.path.join(model_dir, "pka_basic_regression.pkl"))
        self.feature_type = feature_type

    def _load_model(self, clf_modelpath, acidic_modelpath, basic_modelpath):
        # Load regression boosters properly into separate Booster instances
        self.acidic_reg = xgb.Booster()
        self.acidic_reg.load_model(acidic_modelpath)
        self.basic_reg = xgb.Booster()
        self.basic_reg.load_model(basic_modelpath)

        # Load classifier ensemble (could be sklearn-style estimators or xgboost Booster objects)
        bundle = joblib.load(clf_modelpath)
        self.clf = bundle["models"]

    def predict(self, mols):
        mols_features = DataUtils.get_molecular_features(mols, self.feature_type)
        # Some saved classifiers may be xgboost.Booster objects which expect a DMatrix;
        # create a DMatrix once and use it when needed.
        try:
            dmat = xgb.DMatrix(mols_features)
        except Exception:
            dmat = None

        preds = []
        for clf in self.clf:
            # If classifier is a Booster, use DMatrix for prediction
            if isinstance(clf, xgb.Booster):
                if dmat is None:
                    preds.append(clf.predict(xgb.DMatrix(mols_features)))
                else:
                    preds.append(clf.predict(dmat))
            else:
                # sklearn-like estimator or xgboost sklearn wrapper
                preds.append(clf.predict(mols_features))

        clf_labels = np.array(preds).T

        # Regression boosters expect DMatrix as input
        if isinstance(self.acidic_reg, xgb.Booster):
            acidic_scores = self.acidic_reg.predict(dmat if dmat is not None else xgb.DMatrix(mols_features))
        else:
            acidic_scores = self.acidic_reg.predict(mols_features)

        if isinstance(self.basic_reg, xgb.Booster):
            basic_scores = self.basic_reg.predict(dmat if dmat is not None else xgb.DMatrix(mols_features))
        else:
            basic_scores = self.basic_reg.predict(mols_features)
        rets = []
        for idx, clf_label in enumerate(clf_labels):
            ret = {}
            if clf_label[0] == 1: ret["acidic"] = acidic_scores[idx]
            if clf_label[1] == 1: ret["basic"] = basic_scores[idx]
            rets.append(ret)
        return rets

