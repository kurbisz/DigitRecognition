import pandas as pd
import tensorflow_decision_forests as tfdf
from sklearn.datasets import fetch_openml

PREFIX = "[INFO] "


def main():
    print(PREFIX, "LOADING DATA...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")

    print(PREFIX, "PARSING TO DATAFRAME...")
    data = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)
    data["label"] = [str(int(x) - 1) for x in mnist.target]

    train_ds_pd = data.head(60000)
    test_ds_pd = data.tail(10000)

    label = "label"

    print(PREFIX, "PARSING TO TF DATASET...")
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

    model = tfdf.keras.RandomForestModel(
        verbose=1,
        max_depth=32,
        num_trees=400,
        random_seed=420,
    )

    print(PREFIX, "TRAINING MODEL...")
    model.fit(train_ds)

    model.compile(metrics=["accuracy"])
    evaluation = model.evaluate(test_ds, return_dict=True)
    print()

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
