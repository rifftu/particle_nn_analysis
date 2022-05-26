import numpy as np
from sklearn.utils import shuffle

class Jets_dataset:

    ORDER_OF_IMPORTANCE = np.array([3,5,0,9,6,7,8,1,4,2])

    def __init__(self, bg_raw_data: np.ndarray, sig_raw_data: np.ndarray) -> None:
        bg_sorted_parameters = bg_raw_data[:,:,self.ORDER_OF_IMPORTANCE]
        sig_sorted_parameters = sig_raw_data[:,:,self.ORDER_OF_IMPORTANCE]
        self.bg = bg_sorted_parameters.reshape((-1, 100))
        self.sig = sig_sorted_parameters.reshape((-1, 100))
        assert len(self.bg) == len(bg_raw_data)
        assert len(self.sig) == len(sig_raw_data)
        print(f"bg:{self.bg.shape}, sig:{self.sig.shape}")

    @staticmethod
    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        y = y.astype(int)
        return np.eye(num_classes, dtype='uint8')[y]

    def generate_dataset(self,  sig_noise_ratio: float, num_cols: int = 100,num_rows:int = 20000, val_size:int = 5000):
        # keep a counter when using
        bg_access_counter = 0
        sig_access_counter = 0

        # do the positive ones
        num_real_pos = int(sig_noise_ratio * num_rows)
        num_fake_pos = num_rows - num_real_pos

        X_pos_real = self.sig[sig_access_counter:sig_access_counter + num_real_pos, :num_cols]
        sig_access_counter += num_real_pos
        X_pos_fake = self.bg[bg_access_counter:bg_access_counter + num_fake_pos, :num_cols]
        bg_access_counter += num_fake_pos

        X_pos_train = np.concatenate((X_pos_fake, X_pos_real))
        y_pos_train = np.ones(len(X_pos_train))

        # do the negative ones
        X_neg_train = self.bg[bg_access_counter:bg_access_counter + num_rows, :num_cols]
        bg_access_counter += num_rows
        y_neg_train = np.zeros(len(X_neg_train))

        # finalize training set
        X_train = np.concatenate((X_pos_train, X_neg_train))
        y_train = np.concatenate((y_pos_train, y_neg_train))
        X_train, y_train = shuffle(X_train, y_train)

        # make validation set
        X_pos_val = self.sig[sig_access_counter:sig_access_counter + val_size, :num_cols]
        sig_access_counter += val_size
        X_neg_val = self.bg[bg_access_counter:bg_access_counter + val_size, :num_cols]
        bg_access_counter += val_size

        X_val = np.concatenate((X_pos_val, X_neg_val))
        y_val = np.concatenate((np.ones(val_size), np.zeros(val_size)))
        X_val, y_val = shuffle(X_val, y_val)


        print("X_train,y_train,X_val,y_val")
        return (X_train, self.to_categorical(y_train, 2),X_val,self.to_categorical(y_val,2))
