import torch

class Trainer:
    def __init__(self, model=None, max_epochs=10):
        self.model = model.model
        self.max_epochs = max_epochs
        self.optim = self.configure_optimizers()
        print ("Trainer initialized with the model:") 
        print(self.model)

    def training_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def predict(self, features):
        self.model.eval()
        return self.model(features)

    def validation_step(self, batch):
        self.model.eval()
        return self.training_step(batch)

    def log(self, epoch, losse):
        mean_loss = torch.tensor(losse).mean()  # Mean loss
        print(f'Epoch {epoch:3d}/{self.max_epochs:d}: loss = {mean_loss.item():4f}')

    def fit(self, data, log_n=100):
        self.model.train()
        # Record zeroth loss from first minibatch
        mini_batch = next(iter(data))
        self.log(0,self.fit_epoch(mini_batch).item())
        
        # Loop over epochs
        for epoch in range(1,self.max_epochs+1):
            losse = []  # Loss per epoch
            for mini_batch in data:

                # Minibatch construct
                #mini_batch = next(iter(data))
                self.optim.lr = 0.1 if epoch < self.max_epochs/2 else 0.01

                # Fit the epoch and record loss
                loss = self.fit_epoch(mini_batch)

                # Log loss
                losse.append(loss.item())
                #if epoch % log_n == 0:
            self.log(epoch, losse)

    def fit_epoch(self, mini_batch):

        # Loss comes from forward pass/training step
        loss = self.training_step(mini_batch)

        # Backward pass
        self.optim.zero_grad()
        loss.backward()

        # Update using optimizer
        self.optim.step()

        return loss


