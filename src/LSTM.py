import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Tuple, List, Callable, Optional, Any


class LSTM(nn.Module):
    """
    LSTM model

    :attr:`input_size`: the size of the input
    :attr:`output_size`: the size of the output
    :attr:`lstm_num_layers`: the number of LSTM layers
    :attr:`lstm_hidden_size`: the size of the hidden state of the LSTM
    :attr:`num_layers`: the number of layers in the model
    :attr:`hidden_size`: the size of the hidden state of the model
    :attr:`device`: the device to use
    :attr:`l_lstm`: the LSTM layer
    :attr:`l_lnrs`: the linear layers
    """

    def __init__(self, input_size, output_size, **kwargs) -> None:
        """
        Initialize the LSTM model

        :param input_size: the size of the input
        :param output_size: the size of the output
        :param kwargs: the keyword arguments

        :return: None
        """

        super(LSTM, self).__init__()

        self.lstm_num_layers = kwargs.get("lstm_num_layers", 1)
        self.lstm_hidden_size = kwargs.get("lstm_hidden_size", 1)

        self.num_layers = kwargs.get("num_layers", 1)
        self.hidden_size = kwargs.get("hidden_size", 1)

        self.device = kwargs.get("device", torch.device("cpu"))

        self.l_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            device=self.device
        )

        def create_layers() -> List[nn.Linear]:
            """
            Create the linear layers

            :return: the linear layers
            """

            layers = []

            for _ in range(self.num_layers):
                layers += [
                    nn.Linear(self.hidden_size, self.hidden_size, device=self.device),
                    nn.ReLU(),
                    nn.Dropout(p=kwargs.get("dropout", 0.2)),
                ]

            return layers

        self.l_lnrs = nn.Sequential(
            nn.ReLU(), # for the hidden_state

            nn.Linear(self.lstm_hidden_size, self.hidden_size, device=self.device), # To receive the output from the LSTM
            nn.ReLU(),

            *create_layers(),

            nn.Linear(self.hidden_size, output_size, device=self.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        :param x: tensor of shape (B, N, X) where B is the batch_size, N the sequence and X the data itself

        :return: tensor of shape (B, N, O) where O is the output_size
        """

        h_0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=self.device)
        c_0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=self.device)

        _, (hn, _) = self.l_lstm(x, (h_0, c_0))

        hn = hn[-1].view(-1, self.lstm_hidden_size)

        y = self.l_lnrs(hn)

        return y


def train_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[List[Callable[[Any], None]]]=None,
    ) -> Tuple[float, float]:
    """
    Train the model on the training set.

    :param model:      The model to train.
    :param dataloader: The training dataloader.
    :param criterion:  The criterion to use.
    :param optimizer:  The optimizer to use.
    :param callbacks:  The list of callbacks to use. If None, no callbacks are used.

    :return: the loss and the accuracy of the model on the training set.
    """

    if callbacks is None:
        callbacks = []

    model.train()
    losses = []
    nb_correct = 0

    NB_BATCHES = len(dataloader)
    TOTAL_NB_ELEM = len(dataloader.dataset)

    for (data, label) in dataloader:
        optimizer.zero_grad()

        output = model(data)

        _, predicted = torch.max(output.data, 1)
        nb_correct += (predicted == label).sum().item()

        loss = criterion(output, label)
        loss.backward()

        losses.append(loss.item())

        optimizer.step()

        for callback in callbacks:
            callback(model=model, data=data, label=label, output=output, loss=loss)

    return sum(losses) / NB_BATCHES, nb_correct / TOTAL_NB_ELEM


def test_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        callbacks: Optional[List[Callable[[Any], None]]]=None,
    ) -> Tuple[float, float]:
    """
    Test the model on the test set.

    :param model:      The model to test.
    :param dataloader: The validation or testing dataloader.
    :param criterion:  The criterion to use.
    :param callbacks:  The list of callbacks to use. If None, no callbacks are used.

    :return: the loss and the accuracy of the model on the val/test set.
    """

    if callbacks is None:
        callbacks = []

    model.eval()

    NB_BATCHES = len(dataloader)
    TOTAL_NB_ELEM = len(dataloader.dataset)

    losses = []
    nb_correct = 0

    with torch.no_grad():
        for data, label in dataloader:
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            nb_correct += (predicted == label).sum()

            loss = criterion(output, label)
            losses.append(loss.item())

            for callback in callbacks:
                callback(model=model, data=data, label=label, output=output, loss=loss)

    return sum(losses) / NB_BATCHES, (nb_correct / TOTAL_NB_ELEM).detach().cpu()


def select_device() -> torch.device:
    """
    Return a torch.device selecting GPU if available, cpu otherwise.
    To use multiple GPUs, please write a custom function using nn.DataParallel

    For more informations please refer to https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

    :return: The selected device.
    """

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_parallel(model: nn.Module, devices_ids) -> nn.Module:
    """
    Return a model wrapped in a nn.DataParallel.

    :param model: The model to wrap.
    :param devices_ids: The list of devices to use.

    :return: The model wrapped in a nn.DataParallel.
    """

    return nn.DataParallel(model, device_ids=devices_ids)
