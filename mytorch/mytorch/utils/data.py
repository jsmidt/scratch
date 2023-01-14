import torch

class DataLoader:
    def __init__(self, features, targets, batch_size=64, shuffle=True):
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = True

    def __call__(self):
        # Shuffle indicies
        if self.shuffle == True:
            indices = torch.randperm(self.features.shape[0])
        else:
            indices = torch.arange(self.features.shape[0])
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i: i+self.batch_size]
            yield self.features[batch_indices], self.targets[batch_indices]

    def __iter__(self):
        return self()

    def __len__(self):
        return self.batch_size
