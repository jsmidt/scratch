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

    def fit(self, params, mymodel, train_dataloader, validation_dataloader=None, log_epoch=1, plots=False, plot_log=False):
        '''Train the model by fitting to data'''

        if plots == True:
            self.fig = plt.figure()
            self.axes = plt.gca()
        optimizer = mymodel.configure_optimizers()

        # Print loss from first batch
        batch = train_dataloader[0]
        params, loss = self.fit_step(params, mymodel, optimizer, batch)
        print(
            f'\nEpoch ({0:>5d}/{self.max_epochs:5d}) train_loss = {loss.item():.5g}')

        # Now train over each epoch
        n_epoch = [0]
        n_train_loss = [loss]
        n_val_loss = [loss]
        for epoch in range(1, self.max_epochs+1):
            eloss = []
            for step, batch in enumerate(train_dataloader):

                # Get updated params and loss
                params, loss = self.fit_step(params, mymodel, optimizer, batch)

                # Track loss over epoch
                eloss.append(loss)

            if epoch % log_epoch == 0:
                if step > 10000000:
                    pass
                n_epoch.append(epoch)
                n_train_loss.append(jnp.mean(jnp.array(eloss)).item())
                epoch_str = f'Epoch ({epoch:>5d}/{self.max_epochs:5d}) train_loss = {n_train_loss[-1]:.5f} '

                if validation_dataloader is not None:
                    for bbatch in validation_dataloader:
                        loss = mymodel.validation_step(params, bbatch)
                    n_val_loss.append(loss)
                    epoch_str += f'val_loss = {n_val_loss[-1]:.5f}'

                print(epoch_str)

        if plots == True:
            # Plot output
            if plot_log == True:
                self.axes.semilogy(n_epoch, n_train_loss, 'C0', label='Training, Loss')
                self.axes.semilogy(n_epoch, n_val_loss, 'C1', label='Validation, Loss')
            else:
                self.axes.plot(n_epoch, n_train_loss, 'C0', label='Training, Loss')
                self.axes.plot(n_epoch, n_val_loss, 'C1', label='Validation, Loss')
            self.axes.set_xlabel('Epochs')
            self.axes.set_ylabel('Loss')
            self.axes.legend()
        return params

    def fit_step(self, params, mymodel, optimizer, batch):

        # Get loss and gradients
        loss, grads = jax.value_and_grad(
            mymodel.training_step)(params, batch)

        # Update parameters and return
        params = optimizer.step(params, grads)
        return params, loss
