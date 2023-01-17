import jax
import jax.numpy as jnp
#from functional import mse_loss
import matplotlib.pyplot as plt


def mse_loss(input, target):
    return jnp.mean((input-target)**2)


class Trainer:
    '''Pytorch Lightning like trainer class'''

    def __init__(self, learning_rate=0.001):

        self.learning_rate = learning_rate
        #self.fig = None
        self.fig = plt.figure()
        self.axes = plt.gca()

    def training_step(self, params, model, x, y):
        ''' Loss function for training set'''
        logits = model(params, x)
        loss = mse_loss(logits, y)
        return loss

    def validation_step(self):
        # Loss function for validation set
        # model.eval()
        pass

    def prediction_step(self):
        # prediction
        # model.eval()
        pass

    def configure_optimzers(self):
        # optimizers to be used
        pass

    def fit(self, params, model, dataloader, max_epochs=100, log_epoch=1):
        '''Train the model by fitting to data'''

        #if self.fig is None:

        
        # Print loss from first batch
        x, y = dataloader[0]
        params, loss = self.fit_step(params, model, x, y)
        print(f'\nEpoch ({0:>5d}/{max_epochs:5d}) train_loss = {loss.item():.5g}')

        # Now train over each epoch
        n_epoch = [0]
        n_loss = [loss]
        for epoch in range(1, max_epochs+1):
            eloss = []
            for batch, (x, y) in enumerate(dataloader):

                # Get updated params and loss
                params, loss = self.fit_step(params, model, x, y)
                
                # Track loss over epoch
                eloss.append(loss)
            
            if epoch % log_epoch == 0:
                n_epoch.append(epoch)
                n_loss.append(jnp.mean(jnp.array(eloss)).item())
                print(f'Epoch ({epoch:>5d}/{max_epochs:5d}) train_loss = {n_loss[-1]:.5g}')
                

        # Plot output
        self.axes.plot(n_epoch, n_loss,'C0', label='Training, Loss')
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Loss')
        self.axes.legend()
        return params

    def fit_step(self, params, model, x, y):

        # Get loss and gradients
        loss, grads = jax.value_and_grad(self.training_step)(params, model, x, y)

        # Update parameters and return
        params = self.optimizer(params, grads)
        return params, loss

    def optimizer(self, params, grads):
        params = jax.tree_map(
            lambda p, g: p - self.learning_rate * g, params, grads)
        return params


