import torch


class DataLoader:
    def __init__(self, features, targets, batch_size=64, shuffle=True):
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle == True:
            self.indices = torch.randperm(self.features.shape[0])
        else:
            self.indices = torch.arange(self.features.shape[0])

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i: i+self.batch_size]
            yield self.features[batch_indices], self.targets[batch_indices]

    def __len__(self):
        return len(range(0, len(self.indices), self.batch_size))
