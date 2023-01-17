import jax
import jax.numpy as jnp
# from functional import mse_loss
import matplotlib.pyplot as plt


def mse_loss(input, target):
    return jnp.mean((input-target)**2)


class Trainer:
    '''Pytorch Lightning like trainer class'''

    def __init__(self, max_epochs=100):

        # self.learning_rate = learning_rate
        # self.fig = None
        self.max_epochs = max_epochs

    def fit(self, params, mymodel, dataloader, log_epoch=1):
        '''Train the model by fitting to data'''

        #self.fig = plt.figure()
        #self.axes = plt.gca()
        optimizer = mymodel.configure_optimizers()

        # Print loss from first batch
        batch = dataloader[0]
        params, loss = self.fit_step(params, mymodel, optimizer, batch)
        print(
            f'\nEpoch ({0:>5d}/{self.max_epochs:5d}) train_loss = {loss.item():.5g}')

        # Now train over each epoch
        n_epoch = [0]
        n_loss = [loss]
        for epoch in range(1, self.max_epochs+1):
            eloss = []
            for step, batch in enumerate(dataloader):

                # Get updated params and loss
                params, loss = self.fit_step(params, mymodel, optimizer, batch)

                # Track loss over epoch
                eloss.append(loss)

            if epoch % log_epoch == 0:
                if step > 100000000:
                    pass
                n_epoch.append(epoch)
                n_loss.append(jnp.mean(jnp.array(eloss)).item())
                print(
                    f'Epoch ({epoch:>5d}/{self.max_epochs:5d}) train_loss = {n_loss[-1]:.5g}')

        # Plot output
        #self.axes.plot(n_epoch, n_loss, 'C0', label='Training, Loss')
        #self.axes.set_xlabel('Epochs')
        #self.axes.set_ylabel('Loss')
        #self.axes.legend()
        return params

    def fit_step(self, params, mymodel, optimizer, batch):

        # Get loss and gradients
        loss, grads = jax.value_and_grad(
            mymodel.training_step)(params, batch)

        # Update parameters and return
        params = optimizer.step(params, grads)
        return params, loss
