# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    # (1) Your code here; theta = ...
    theta = torch.linspace(-math.pi, math.pi, steps=1000)
    assert theta.shape == (1000,)

    # (2) Your code here; rho = ...
    rho = (torch.ones_like(theta) + 0.9*torch.cos(8.0*theta)) * (torch.ones_like(theta) + 0.1*torch.cos(24.0*theta)) * (0.9*torch.ones_like(theta) + 0.05*torch.cos(200.0*theta)) * (torch.ones_like(theta) + torch.sin(theta))
    assert torch.is_same_size(rho, theta)
    
    # (3) Your code here; x = ...
    # (3) Your code here; y = ...
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    # Your code here
    # Count neighbours for each cell with convolution
    num_alive_neighbors = torch.zeros_like(alive_map)
    h, w = alive_map.shape
    
    kernel = torch.LongTensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    num_alive_neighbors = torch.conv2d(alive_map.view(1, 1, h, w), kernel.view(1, 1, 3, 3), padding=1)
    
    # Apply game rules
    new_alive_map = torch.empty_like(alive_map)
    born = torch.eq(num_alive_neighbors, 3) & torch.eq(alive_map, 0)
    survived = (torch.eq(num_alive_neighbors, 2) | torch.eq(num_alive_neighbors, 3)) & torch.eq(alive_map, 1)
    new_alive_map = born | survived
    
    # Output the result
    alive_map.copy_(new_alive_map.view(h, w))

""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
class NeuralNet:
    def __init__(self):
        # Your code here

        self.input_size = 28*28
        self.hidden_size = 50
        self.output_size = 10

        self.mu = 0
        self.sigma = 0.01

        self.learning_rate = 10e-3
        self.n_epoch = 10
        self.batch_size = 128
        self.eps = 1e-10

        self.W1 = torch.empty(self.input_size, self.hidden_size).normal_(self.mu, self.sigma).requires_grad_(True)
        self.b1 = torch.empty(self.hidden_size).normal_(self.mu, self.sigma).requires_grad_(True)
        self.W2 = torch.empty(self.hidden_size, self.output_size).normal_(self.mu, self.sigma).requires_grad_(True)
        self.b2 = torch.empty(self.output_size).normal_(self.mu, self.sigma).requires_grad_(True)

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        # Your code here
        
        z = images.view(-1, 28*28)
        z = torch.mm(z, self.W1) + self.b1
        z = torch.relu(z)
        z = torch.mm(z, self.W2) + self.b2
        return z.softmax(dim=1)

def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """
    # Your code here
    
    accuracy = (torch.sum(torch.LongTensor(model.predict(images).argmax(dim=1)) == labels)).item() / labels.size(0)
    return accuracy

def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    # Your code here
    
    loss_history = []
    y_train_hot = torch.Tensor(torch.eye(10)[y_train])

    for i in range(model.n_epoch):
        for x_batch, y_batch in get_batches((X_train, y_train_hot), model.batch_size):

            # zero_grad
            model.W1.grad = torch.zeros_like(model.W1)
            model.b1.grad = torch.zeros_like(model.b1)
            model.W2.grad = torch.zeros_like(model.W2)
            model.b2.grad = torch.zeros_like(model.b2)

            # forward
            predictions = model.predict(x_batch)
            loss = -torch.sum(torch.log(torch.clamp(predictions, model.eps, 1 - model.eps)) * y_batch) / y_batch.size(0)

            # backward
            loss.backward()

            with torch.no_grad():
                model.W1 -= model.learning_rate * model.W1.grad
                model.b1 -= model.learning_rate * model.b1.grad
                model.W2 -= model.learning_rate * model.W2.grad
                model.b2 -= model.learning_rate * model.b2.grad 

            loss_history.append(loss.detach().item())
    
# from HW1 transferred to torch
def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = torch.arange(n_samples)
    indices = indices[torch.randperm(indices.size()[0])]
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]
