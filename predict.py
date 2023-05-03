# predict.py
import joblib
import pandas as pd
import numpy as np

model = joblib.load("logistic_regression.joblib")


def make_prediction(inputs):
    inputs_df = pd.DataFrame(
        np.array([inputs]),
        columns=["sepal length ", "sepal width ", "petal length ", "petal width "],
    )
    predictions = model.predict(inputs_df)
    return predictions