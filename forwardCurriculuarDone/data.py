import numpy as np
import tensorflow as tf
import torch
from pennylane import numpy as pnp
from sklearn.decomposition import PCA


def data_load_and_process(dataset="mnist", reduction_sz: int = 4, train_len=400, test_len=100):
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == "kmnist":
        # Path to training images and corresponding labels provided as numpy arrays
        kmnist_train_images_path = f"{data_path}/kmnist-train-imgs.npz"
        kmnist_train_labels_path = f"{data_path}/kmnist-train-labels.npz"

        # Path to the test images and corresponding labels
        kmnist_test_images_path = f"{data_path}/kmnist-test-imgs.npz"
        kmnist_test_labels_path = f"{data_path}/kmnist-test-labels.npz"

        x_train = pnp.load(kmnist_train_images_path)["arr_0"]
        y_train = pnp.load(kmnist_train_labels_path)["arr_0"]

        # Load the test data from the corresponding npz files
        x_test = pnp.load(kmnist_test_images_path)["arr_0"]
        y_test = pnp.load(kmnist_test_labels_path)["arr_0"]

    x_train, x_test = (
        x_train[..., pnp.newaxis] / 255.0,
        x_test[..., pnp.newaxis] / 255.0,
    )
    train_filter_tf = pnp.where((y_train == 0) | (y_train == 1))
    test_filter_tf = pnp.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
    x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
    x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

    X_train = PCA(reduction_sz).fit_transform(x_train)
    X_test = PCA(reduction_sz).fit_transform(x_test)
    x_train, x_test = [], []
    for x in X_train:
        x = (x - x.min()) * (2 * pnp.pi / (x.max() - x.min()))
        x_train.append(x)
    for x in X_test:
        x = (x - x.min()) * (2 * pnp.pi / (x.max() - x.min()))
        x_test.append(x)
    return x_train[:train_len], x_test[:test_len], y_train[:train_len], y_test[:test_len]


def new_data(batch_sz, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_sz):
        n, m = pnp.random.randint(len(X)), pnp.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)

    # X1_new 처리
    X1_new_array = pnp.array(X1_new)
    X1_new_tensor = torch.from_numpy(X1_new_array).float()

    # X2_new 처리
    X2_new_array = pnp.array(X2_new)
    X2_new_tensor = torch.from_numpy(X2_new_array).float()

    # Y_new 처리
    Y_new_array = pnp.array(Y_new)
    Y_new_tensor = torch.from_numpy(Y_new_array).float()
    return X1_new_tensor, X2_new_tensor, Y_new_tensor


def new_triplet_data(batch_sz, X, Y):
    triplets = []
    for _ in range(batch_sz):
        anchor_idx = np.random.randint(len(X))
        anchor_x, anchor_y = X[anchor_idx], Y[anchor_idx]

        # positive: 같은 클래스
        pos_candidates = [i for i, y in enumerate(Y) if y == anchor_y and i != anchor_idx]
        positive_idx = np.random.choice(pos_candidates)

        # negative: 다른 클래스
        neg_candidates = [i for i, y in enumerate(Y) if y != anchor_y]
        negative_idx = np.random.choice(neg_candidates)

        triplets.append((X[anchor_idx], X[positive_idx], X[negative_idx]))

    Xa, Xp, Xn = zip(*triplets)
    Xa_tensor = torch.from_numpy(np.array(Xa)).float()
    Xp_tensor = torch.from_numpy(np.array(Xp)).float()
    Xn_tensor = torch.from_numpy(np.array(Xn)).float()
    return Xa_tensor, Xp_tensor, Xn_tensor


def get_class_balanced_batch(X, Y, dc=16):
    """클래스별로 dc개씩 균등하게 뽑은 batch 반환"""
    Xa, Ya = [], []
    for cls in [0, 1]:
        indices = [i for i, y in enumerate(Y) if y == cls]
        selected = np.random.choice(indices, size=dc, replace=False)
        for i in selected:
            Xa.append(X[i])
            Ya.append(Y[i])
    Xa = np.array(Xa)
    Ya = np.array(Ya)
    return torch.from_numpy(Xa).float(), torch.from_numpy(Ya).int()