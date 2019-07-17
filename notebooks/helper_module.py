'''Module containing function that are too large to be included in the notebooks.'''

import torch

class SimpleMLP(torch.nn.Module):
    def __init__(self, n_inputs=1, n_outputs=1, n_layers=1, n_neurons=10, activation=torch.sigmoid, batch_norm=False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.batch_norm = batch_norm
        self.layers = torch.nn.ModuleList()
        if self.batch_norm:
            # input layer to first hidden layer
            self.layers.append(torch.nn.Linear(self.n_inputs, self.n_neurons*2, bias=False))
            self.layers.append(torch.nn.BatchNorm1d(self.n_neurons*2))
            # add more hidden layers if specified
            if self.n_layers > 2:
                for hidden in range(self.n_layers-2):
                    self.layers.append(torch.nn.Linear(self.n_neurons*2, self.n_neurons*2, bias=False))
                    self.layers.append(torch.nn.BatchNorm1d(self.n_neurons*2))
            self.layers.append(torch.nn.Linear(self.n_neurons*2, self.n_neurons, bias=False))
            self.layers.append(torch.nn.BatchNorm1d(self.n_neurons))
        else:
            # input layer to first hidden layer
            self.layers.append(torch.nn.Linear(self.n_inputs, self.n_neurons))
            # add more hidden layers if specified
            if self.n_layers > 1:
                for hidden in range(self.n_layers-1):
                    self.layers.append(torch.nn.Linear(self.n_neurons, self.n_neurons))
        # last hidden layer to output layer
        self.layers.append(torch.nn.Linear(self.n_neurons, self.n_outputs))
        print("Created model with {} weights.".format(self.model_parameters()))


    def forward(self, x):
        if self.batch_norm:
            for i_layer in range(len(self.layers)-1):
                if isinstance(self.layers[i_layer], torch.nn.Linear):
                    x = self.layers[i_layer](x)
                else:
                    x = self.activation(self.layers[i_layer](x))
        else:
            for i_layer in range(len(self.layers)-1):
                x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)


    def model_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def approximate_function(x_train, y_train, x_val, y_val, model, l_rate=0.001, batch_size=128,
                         max_iter=1000, path=None, verbose=100):
    '''Train MLP to approximate a function y(x).
       The training stops when the maximum number of training epochs is reached.

    Parameters
    ----------
    x_train - array-like : argument of the function; used for training
    y_train - array-like : function value at x; used for training
    x_val - array-like   : argument of the function; used for validation
    y_val - array-like   : function value at x; used for validation
    model - SimpleMLP    : PyTorch model which is adjusted to approximate the function
    l_rate - Float       : learning rate for weight optimization
    batch_size - Integer : batch size for training data
    max_iter - Integer   : maximum number of allowed training epochs
    path - String        : location to save model weights
    verbose - Integer    : defines frequency for loss information output

    Returns
    -------
    model - SimpleMLP       : trained version of the given model
    loss_train - array-like : training loss
    loss_val - array-like   : validation loss

    '''
    # convert numpy arrays to torch tensors
    x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
    x_val_tensor = torch.from_numpy(x_val.astype(np.float32))
    y_val_tensor = torch.from_numpy(y_val.astype(np.float32))
    # define loss function
    criterion = torch.nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=l_rate)

    # training loop
    history_train = []
    history_val = []
    best_loss = 1.0E5
    count = 0
    # move model and data to gpu if available
    model.to(device)
    x_val_device, y_val_device = x_val_tensor.to(device), y_val_tensor.to(device)
    n_batches = np.ceil(x_train.shape[0] / batch_size)

    for e in range(1, max_iter+1):
        # backpropagation
        model = model.train()
        loss_sum_batches = 0.0
        for b in range(int(n_batches)):
            x_batch = x_train_tensor[b*batch_size:min(x_train_tensor.shape[0], (b+1)*batch_size)].to(device)
            y_batch = y_train_tensor[b*batch_size:min(x_train_tensor.shape[0], (b+1)*batch_size)].to(device)
            optimizer.zero_grad()
            output_train = model(x_batch)
            loss_train = criterion(output_train.squeeze(dim=1), y_batch)
            loss_train.backward()
            optimizer.step()
            loss_sum_batches += loss_train.item()
        history_train.append(loss_sum_batches / n_batches)

        # validation
        with torch.no_grad():
            model = model.eval()
            output_val = model.forward(x_val_device)
            loss_val = criterion(output_val.squeeze(dim=1), y_val_device)
            history_val.append(loss_val.item())

        # check maximum error for validation data
        diff_val = output_val.squeeze(dim=1) - y_val_device
        max_diff_val = np.amax(np.absolute(diff_val.cpu().detach().numpy()))
        if history_train[-1] < best_loss:
            count += 1
            best_loss = history_train[-1]
            if count % verbose == 0:
                print("Training loss decreased in epoch {}: {}".format(e, history_train[-1]))
                print("Validation loss/max. dev.: {}/{}".format(loss_val.item(), max_diff_val))
                print("--------------------------------")
            if path is not None:
                if count % verbose == 0: print("Saving model as {}".format(path))
                torch.save(model.state_dict(), path)
    return model.eval(), np.asarray(history_train), np.asarray(history_val)
