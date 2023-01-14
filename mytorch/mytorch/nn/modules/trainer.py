import torch

class Trainer:
    def __init__(self, model=None, max_epochs=100):
        self.model = model.model
        self.max_epochs = max_epochs
        self.optim = self.configure_optimizers()
        print ("Trainer initialized with the model:") 
        print(self.model)

    def training_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def eval(self, batch):
        return self.validation_step(batch)

    def validation_step(self, batch):
        return self.training_step(batch)

    def log(self, epoch, lossi, log_n):
        mean_loss = torch.tensor(lossi)[-log_n:].mean()
        print(f'Epoch {epoch:7d}/{self.max_epochs:d}: loss = {mean_loss.item():4f}')

    def fit(self, data, log_n=10000):
        lossi = []
        ud = []
        for epoch in range(self.max_epochs):

            # Minibatch construct
            mini_batch = next(iter(data))
            self.optim.lr = 0.1 if epoch < self.max_epochs/2 else 0.01

            # Fit the epoch and record loss
            loss = self.fit_epoch(mini_batch)

            # Log loss
            lossi.append(loss.item())
            if epoch % log_n == 0:
                self.log(epoch, lossi, log_n)

    def fit_epoch(self, mini_batch):

        # Loss comes from forward pass/training step
        loss = self.training_step(mini_batch)

        # Backward pass
        self.optim.zero_grad()
        loss.backward()

        # Update using optimizer
        self.optim.step()

        return loss
