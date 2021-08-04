import pickle
from io import StringIO
from os.path import join

import pandas as pd  # type: ignore
from interpret.glassbox import ExplainableBoostingClassifier  # type: ignore

from preprocess import oversample, preprocess

random_state = 42
target_columns = ["incident"]


def train():
    df = preprocess()

    # our ExplainableBoostingClassifier requires a balanced training dataset. We can acheive this by oversampling. To speed it up we train on only N samples with the positive and negative incidents, with replacement.
    df = oversample(df, "incident", n=15000)

    y_train = df[target_columns]
    X_train = df.drop(columns=target_columns)

    # we use a glassbox model, and put interactions=0 to avoid combining features
    # see https://interpret.ml/docs/getting-started#train-a-glassbox-model
    model = ExplainableBoostingClassifier(random_state=random_state, interactions=0)
    print(model.fit(X_train, y_train))

    save_model(model, "models")


def save_model(model, model_dir):
    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)


def model_fn(model_dir):
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data):
    input = pd.read_csv(StringIO(input_data), index_col=0, parse_dates=True)
    return input


if __name__ == "__main__":
    train("data/public.csv")
