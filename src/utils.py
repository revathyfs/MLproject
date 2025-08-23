import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """Save any Python object using dill"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple models with GridSearchCV.
    Returns:
        report (dict): Model performance details
        best_model_name (str): Name of the best model
        best_model_obj (object): Trained best model
    """
    try:
        report = {}
        best_model_name = None
        best_model_score = float("-inf")
        best_model_obj = None

        for model_name, model in models.items():
            para = param.get(model_name, {})

            # Hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Retrain with best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store results
            report[model_name] = {
                "train_r2": train_model_score,
                "test_r2": test_model_score,
                "best_params": gs.best_params_
            }

            # Track best model
            if test_model_score > best_model_score:
                best_model_score = test_model_score
                best_model_name = model_name
                best_model_obj = model

        return report, best_model_name, best_model_obj

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a Python object using dill"""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
