import jax
import jax.numpy as jnp

class Dataset:
    '''Takes x, y (features, targets) and puts them in a dataset'''

    def __init__(self, features, targets, dtype_features=jnp.float32, dtype_labels=jnp.float32) -> None:
        self.features = jnp.array(features, dtype=dtype_features)
        self.targets = jnp.array(targets, dtype=dtype_labels)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[jnp.array, jnp.array]:
        return self.features[idx], self.targets[idx]


class DataLoader:
    '''Makes dataset iterable and randomly shuffled if a key supplied'''

    def __init__(self, dataset, key=None, batch_size=32) -> None:

        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key

    def __iter__(self):

        # If key present, randomly shuffle
        if self.key is not None:
            # After looping through data in epoch, reorder data for next epoch
            self.key, i_key = jax.random.split(self.key)
            self.indices = jax.random.permutation(i_key, len(self.dataset))
        else:
            self.indices = jnp.arange(len(self.dataset))

        # Loop over indicies and yield result
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i: i+self.batch_size]
            yield self.dataset[batch_indices]

    def __len__(self):
        return len(range(0, len(self.dataset), self.batch_size))

    def __getitem__(self, idx: int) -> tuple[jnp.array, jnp.array]:
        return self.dataset[idx]

def random_split(dataset, train_length=0.9, key=None):
    # Get length of dataset
    len_data = len(dataset)

    # Get length of desired train data
    if train_length <= 1.0:
        n_train = int(train_length*len_data)
    else:
        n_train = train_length

    # Two sets of depending on whether random key present
    if key is not None:
        key, i_key = jax.random.split(key)
        indices = jax.random.permutation(i_key, len(dataset))
        train_indicies = indices[:n_train]
        val_indicies = indices[n_train:]
    else:
        indices = jnp.arange(len(dataset))
        train_indicies = indices[:n_train]
        val_indicies = indices[n_train:]

    train_X, train_y = dataset[train_indicies]
    val_X, val_y = dataset[val_indicies]

    train_data = Dataset(train_X, train_y)
    val_data = Dataset(val_X, val_y)

    return train_data, val_data

