import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def scoring_fn(y, y_pred):
    logger.info("scoring_fn")
    return roc_auc_score(y, y_pred)


if __name__ == "__main__":
    """Scoring Function

    This function is called by Unearthed's SageMaker pipeline. It must be left intact.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--predicted",
        type=str,
        default="/opt/ml/processing/input/predictions/public.csv.gz",
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/scores/public.txt"
    )
    args = parser.parse_args()

    # read the data file
    df_actual = pd.read_csv(args.actual)

    # recreate the targets
    target_columns = ["incident"]
    targets = df_actual[target_columns]
    logger.info(f"true targets have shape of {targets.shape}")

    # read the predictions
    df_pred = pd.read_csv(args.predicted, header=None)
    logger.info(f"predictions have shape of {df_pred.shape}")

    score = scoring_fn(targets, df_pred)

    # write to the output location
    with open(args.output, "w") as f:
        f.write(str(score))

