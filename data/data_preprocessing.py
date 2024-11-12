# preprocesses the data for the training and testing of the model

# indices description:
# 0: pt, 1: eta, 2: phi, 3: e, 4: muonsInJet, 5: btag, 6: recoPt, 7: recoPhi, 8: recoEta, 9: muonsInJet, 10: recoNConstituents, 11: nef, 12: nhf, 13: cef, 14: chf, 15: qgl, 16: jetId, 17: ncharged, 18: nneutral, 19: ctag, 20: nSV

import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

path = os.path.dirname(__file__)
MEAN_X_PATH = os.path.join(path, "mean_x.npy")
SCALE_X_PATH = os.path.join(path, "scale_x.npy")
MEAN_Y_PATH = os.path.join(path, "mean_y.npy")
SCALE_Y_PATH = os.path.join(path, "scale_y.npy")

categories = np.array([0, 1, 2, 3, 4, 5, 21]).reshape(-1)


class TrainDataPreprocessor(object):
    def __init__(self, data_kwargs, scaler_x=None, scaler_y=None):
        self.data = np.load(data_kwargs["dataset_path"])
        self.standardize = data_kwargs["standardize"]
        if "flavour_ohe" in data_kwargs:
            self.flavour_ohe = data_kwargs["flavour_ohe"]
        else:
            self.flavour_ohe = False
        self.N_train = data_kwargs["N_train"]
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        print(self.data.shape)
        self.X = self.data[
            : self.N_train, (5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
        ]
        self.Y = self.data[: self.N_train, (0, 1, 2, 3, 4, 9)]
        # pt ratio
        self.X[:, 1] = self.X[:, 1] / self.Y[:, 0]
        # mass ratio
        self.X[:, 15] = self.X[:, 15] / self.Y[:, 3]

        # throw away all NaNs from divisions by 0
        nans_mask = np.isnan(self.X).any(axis=1)
        self.X = self.X[~nans_mask]
        self.Y = self.Y[~nans_mask]

        self.Y[:, 4] = np.abs(self.Y[:, 4])
        # smearing of N constituents to let the network learn the distribution
        self.X[:, 4] = self.X[:, 4] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 4]))
        )
        self.X[:, 11] = self.X[:, 11] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 11]))
        )
        self.X[:, 12] = self.X[:, 12] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 12]))
        )
        self.X[:, 14] = self.X[:, 14] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 14]))
        )

        # ! Flavour Encoding:
        # ! Y indices are:
        # ! 5: 0 (undefined), 6: 1 (u), 7: 2 (d), 8: 3 (s), 9: 4 (c), 10: 5 (b), 11: 21 (g)
        # ! 0: pt, 1: eta, 2: phi, 3: e, 4: muonsInJet
        if self.flavour_ohe:
            encoder = OneHotEncoder(categories=[categories])
            flavour_ohe = encoder.fit_transform(self.Y[:, 4].reshape(-1, 1))
            flavour_ohe = flavour_ohe.toarray()
            self.Y = np.hstack(
                (self.Y[:, 0:4], self.Y[:, 5].reshape(-1, 1), flavour_ohe)
            )

        if self.standardize:
            if (self.scaler_x == None) & (self.scaler_y == None):
                self.scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()
                if (
                    os.path.exists(MEAN_X_PATH)
                    and os.path.exists(SCALE_X_PATH)
                    and os.path.exists(SCALE_Y_PATH)
                    and os.path.exists(MEAN_Y_PATH)
                ):
                    self.scaler_x.mean_ = np.load(MEAN_X_PATH)
                    self.scaler_x.scale_ = np.load(SCALE_X_PATH)
                    self.scaler_y.mean_ = np.load(MEAN_Y_PATH)
                    self.scaler_y.scale_ = np.load(SCALE_Y_PATH)
                else:
                    self.scaler_x.fit(self.X)
                    if self.flavour_ohe:
                        self.scaler_y.fit(self.Y[:, 0:5])
                    else:
                        self.scaler_y.fit(self.Y)
                    print("train saving")
                    np.save(MEAN_X_PATH, self.scaler_x.mean_)
                    np.save(SCALE_X_PATH, self.scaler_x.scale_)
                    np.save(MEAN_Y_PATH, self.scaler_y.mean_)
                    np.save(SCALE_Y_PATH, self.scaler_y.scale_)
            self.X = self.scaler_x.transform(self.X)
            if self.flavour_ohe:
                self.Y[:, 0:5] = self.scaler_y.transform(self.Y[:, 0:5])
            else:
                self.Y = self.scaler_y.transform(self.Y)

    def get_scaler(self):
        return self.scaler

    def __len__(self):
        return len(self.data)

    def get_dataset(self):
        return self.X, self.Y

    def invert_standardize(self, X, Y):
        X = self.scaler_x.inverse_transform(X)
        if self.flavour_ohe:
            Y[:, 0:5] = self.scaler_y.inverse_transform(Y[:, 0:5])
        else:
            Y = self.scaler_y.inverse_transform(Y)
        return X, Y


class TestDataPreprocessor(object):
    def __init__(self, data_kwargs, scaler_x=None, scaler_y=None, allow_nans=False):
        self.data = np.load(data_kwargs["dataset_path"])
        self.standardize = data_kwargs["standardize"]
        if "flavour_ohe" in data_kwargs:
            self.flavour_ohe = data_kwargs["flavour_ohe"]
        else:
            self.flavour_ohe = False
        self.N_train = data_kwargs["N_train"]
        self.N_test = data_kwargs["N_test"]
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        # print(self.data.shape)
        self.X = self.data[
            self.N_train : self.N_train + self.N_test,
            (5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
        ]
        self.Y = self.data[
            self.N_train : self.N_train + self.N_test, (0, 1, 2, 3, 4, 9)
        ]
        # pt ratio
        self.X[:, 1] = self.X[:, 1] / self.Y[:, 0]
        # mass ratio
        self.X[:, 15] = self.X[:, 15] / self.Y[:, 3]

        if not allow_nans:
            # throw away all NaNs from divisions by 0
            nans_mask = np.isnan(self.X).any(axis=1)
            self.X = self.X[~nans_mask]
            self.Y = self.Y[~nans_mask]

        self.Y[:, 4] = np.abs(self.Y[:, 4])
        # smearing of N constituents to let the network learn the distribution
        self.X[:, 4] = self.X[:, 4] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 4]))
        )
        self.X[:, 11] = self.X[:, 11] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 11]))
        )
        self.X[:, 12] = self.X[:, 12] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 12]))
        )
        self.X[:, 14] = self.X[:, 14] + (
            0.5 * np.random.uniform(-1, 1, len(self.X[:, 14]))
        )

        # ! Flavour Encoding:
        # ! Y indices are:
        # ! 5: 0 (undefined), 6: 1 (u), 7: 2 (d), 8: 3 (s), 9: 4 (c), 10: 5 (b), 11: 21 (g)
        # ! 0: pt, 1: eta, 2: phi, 3: e, 4: muonsInJet
        if self.flavour_ohe:
            encoder = OneHotEncoder(categories=[categories])
            flavour_ohe = encoder.fit_transform(self.Y[:, 4].reshape(-1, 1))
            flavour_ohe = flavour_ohe.toarray()
            self.Y = np.hstack(
                (self.Y[:, 0:4], self.Y[:, 5].reshape(-1, 1), flavour_ohe)
            )

        if self.standardize:
            if (self.scaler_x == None) & (self.scaler_y == None):
                self.scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()
                if (
                    os.path.exists(MEAN_X_PATH)
                    and os.path.exists(SCALE_X_PATH)
                    and os.path.exists(SCALE_Y_PATH)
                    and os.path.exists(MEAN_Y_PATH)
                ):
                    self.scaler_x.mean_ = np.load(MEAN_X_PATH)
                    self.scaler_x.scale_ = np.load(SCALE_X_PATH)
                    self.scaler_y.mean_ = np.load(MEAN_Y_PATH)
                    self.scaler_y.scale_ = np.load(SCALE_Y_PATH)
                else:
                    self.scaler_x.fit(self.X)
                    if self.flavour_ohe:
                        self.scaler_y.fit(self.Y[:, 0:5])
                    else:
                        self.scaler_y.fit(self.Y)
                    print("test saving")
                    np.save(MEAN_X_PATH, self.scaler_x.mean_)
                    np.save(SCALE_X_PATH, self.scaler_x.scale_)
                    np.save(MEAN_Y_PATH, self.scaler_y.mean_)
                    np.save(SCALE_Y_PATH, self.scaler_y.scale_)
            self.X = self.scaler_x.transform(self.X)
            if self.flavour_ohe:
                self.Y[:, 0:5] = self.scaler_y.transform(self.Y[:, 0:5])
            else:
                self.Y = self.scaler_y.transform(self.Y)

    def get_scaler(self):
        return self.scaler


    def __len__(self):
        return len(self.data)

    def get_dataset(self):
        return self.X, self.Y

    def invert_standardize(self, X, Y):
        X = self.scaler_x.inverse_transform(X)
        if self.flavour_ohe:
            print("test invert flavour ohe")
            Y[:, 0:5] = self.scaler_y.inverse_transform(Y[:, 0:5])
        else:
            Y = self.scaler_y.inverse_transform(Y)
        return X, Y


if __name__ == "__main__":
    data_kwargs = {
        "dataset_path": "/scratchnvme/cattafe/gen_ttbar_400k.npy",
        "standardize": True,
        "flavour_ohe": True,
        "N_train": 500000,
    }

    train_dataset = TrainDataPreprocessor(data_kwargs)
    X, Y = train_dataset.get_dataset()
    print(X)
    print(Y)
