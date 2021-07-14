# Don't erase the template code, except "Your code here" comments.

import torch
# Your code here...
import torchvision
import torch.nn as nn
from torch import optim
from datetime import datetime
from tqdm.notebook import tqdm
from torch.nn.modules import loss
from torch.utils.tensorboard import SummaryWriter

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

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.
    
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'
        
    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    
    trans = {'train': torchvision.transforms.Compose([                  
                  # torchvision.transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.06, hue=0.06),
                  torchvision.transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.05),
                  torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                  # torchvision.transforms.RandomGrayscale(),
                  torchvision.transforms.RandomRotation(15),
                  torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.RandomVerticalFlip(),
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

                  ]),

          'test' : torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}
    
    batch_size = 128
    train_data = torchvision.datasets.ImageFolder(path+'train',transform=trans['train'])
    val_data = torchvision.datasets.ImageFolder(path+'val', transform = trans['test'])

    if kind == 'train':
        return torch.utils.data.DataLoader(train_data,
                                          batch_size = batch_size,
                                          num_workers = 2,
                                          shuffle=True)
    if kind == 'val':
        return torch.utils.data.DataLoader(val_data,
                                            batch_size = batch_size,
                                            num_workers = 2,
                                            shuffle=False)

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.
    
    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = torchvision.models.densenet161()
    model = model.to(DEVICE)

    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.
    
    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    
    optimizer = optim.Adam(model.parameters(), lr=10e-4, weight_decay=2e-4)
    return optimizer

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).
    
    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    batch = batch.to(DEVICE)
    output = model(batch)

    return output

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.
    
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    loss = 0
    accuracy = 0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for batch, labels in dataloader: 
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)
        output = model(batch)
        loss += criterion(output, labels).item()/len(labels)
        # print(loss)   
        pred = output.argmax(dim=-1, keepdim=True)
        # print("val_pred:", pred)
        # print("val_true:", labels.view_as(pred))
        # print("accuracy:", pred.eq(labels.view_as(pred)).sum().item())
        accuracy += pred.eq(labels.view_as(pred)).sum().item()/len(labels)
    accuracy *= 100.0
    accuracy /= len(dataloader)
    loss /= len(dataloader)
    return accuracy, loss

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.
    
    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    n_epoch = 40
    # exp_name = datetime.now().isoformat(timespec='seconds')    # exp_name = '2021-04-01T10:11:12'
    # writer = SummaryWriter(log_dir='logs/' + str(exp_name))
    criterion = nn.CrossEntropyLoss()
    
    for i in range(n_epoch):
        model.train()
        train_accuracy = 0
        for batch, labels in tqdm(train_dataloader): 

            batch, labels = batch.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=-1, keepdim=True)
            # print("pred:", pred)
            # print("true:", labels.view_as(pred))

            train_accuracy += pred.eq(labels.view_as(pred)).sum().item()

        print('Train Loss: ', loss.item())
        print('Train Accuracy %: ', 100.0 * train_accuracy / len(train_dataloader.dataset))
        # writer.add_scalar("Train Loss", loss.item(), global_step=i,)
        # writer.add_scalar("Train Accuracy %", 100.0 * train_accuracy / len(train_dataloader.dataset), global_step=i,)
        # torch.save(model.state_dict(), 'drive/MyDrive/DL-HW-2/model_epoch'+str(i)+'.pth')

        val_loss = 0
        val_accuracy = 0
        criterion = nn.CrossEntropyLoss()
        model.eval()
        for batch, labels in tqdm(val_dataloader): 
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            output = model(batch)
            val_loss += criterion(output, labels).item()/len(labels)
            pred = output.argmax(dim=-1, keepdim=True)
            val_accuracy += pred.eq(labels.view_as(pred)).sum().item()/len(labels)
        val_accuracy *= 100.0
        val_accuracy /= len(val_dataloader.dataset)
        val_loss /= len(val_dataloader.dataset)
        # writer.add_scalar("Val Loss", val_loss, global_step=i,)
        # writer.add_scalar("Val Accuracy %", val_accuracy, global_step=i,)

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.
    
    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "747822ca4436819145de8f9e410ca9ca"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"
    
    md5_checksum = "3fb9a9b65765cee94ad6ce5e3d066cb6"
    google_drive_link = "https://drive.google.com/file/d/1ld5Ncad0LQs1U_YWismjtmzAh93UCkqr/view?usp=sharing"
    
    return md5_checksum, google_drive_link
