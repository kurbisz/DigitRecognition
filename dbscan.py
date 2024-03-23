import os

import numpy as np
import tensorflow_datasets as tfds

PREFIX = "[INFO] "


def print_image(data: list[int]) -> None:
    from PIL import Image

    w, h = 28, 28
    new_data = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(28):
        for j in range(28):
            val = 0 if data[i + j * 28] else 255
            new_data[i, j] = [val, val, val]
    img = Image.fromarray(new_data, "RGB")
    img.show()


def DBSCAN(
    data: np.ndarray,
    eps: int,
    min_pts: int,
    am: int,
) -> tuple[np.ndarray, int]:
    act_cluster = 0
    visited = np.zeros(am, dtype=bool)

    # -2 NOISE, -1 NOT ASSIGNED, OTHERS MEANS ID OF CLUSTER
    clusters = np.full(am, -1)

    for nr in range(am):
        if visited[nr]:
            continue
        visited[nr] = True
        neighbours = get_neighbours(data, nr, eps)
        if len(neighbours) < min_pts:
            clusters[nr] = -2
        else:
            new_cluster = act_cluster
            act_cluster += 1
            expand_cluster(
                data,
                nr,
                neighbours,
                new_cluster,
                eps,
                min_pts,
                visited,
                clusters,
            )
    return clusters, act_cluster


def expand_cluster(
    data: np.ndarray,
    nr: int,
    neighbours: set,
    cluster: int,
    eps: int,
    min_pts: int,
    visited: np.ndarray,
    clusters_list: np.ndarray,
) -> None:
    clusters_list[nr] = cluster
    while neighbours:
        neighbour = neighbours.pop()
        if not visited[neighbour]:
            visited[neighbour] = True
            new_neighbours = get_neighbours(data, neighbour, eps)
            if len(new_neighbours) >= min_pts:
                neighbours.update(new_neighbours)
        elif clusters_list[neighbour] < 0:
            clusters_list[neighbour] = cluster


def get_neighbours(data: np.ndarray, nr: int, eps: int) -> set:
    act_data = data[nr]
    dist = np.sum(np.logical_xor(data, act_data), axis=1)
    res2 = np.where(dist < eps)[0]
    return set(res2)


def main():
    print(PREFIX, "LOADING DATA...")
    (ds_train, _), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train_np = tfds.as_numpy(ds_train)
    correct = [ex[1] for ex in ds_train_np]

    am = 1000
    print(PREFIX, "APPLYING SCALING...")
    vectorized_mapping = np.vectorize(lambda x: x >= 32)
    arr = np.array([vectorized_mapping(ex[0].flatten()) for ex in ds_train_np])

    arr = arr[:am, :]

    # experimentally determined - results seems to be the best
    eps, min_pts = 43, 17
    print(PREFIX, eps, min_pts)
    res, clusters = DBSCAN(arr, eps, min_pts, am)
    clusters_list = [[] for _ in range(clusters)]
    noise = 0
    for i in range(len(res)):
        val = res[i]
        if val >= 0:
            clusters_list[val].append(correct[i])
        else:
            noise += 1

    for idx, cluster in enumerate(clusters_list):
        print(PREFIX, f"CLUSTER {idx}:", cluster)

    n = noise / am * 100
    print(PREFIX, "NOISE:", n)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
