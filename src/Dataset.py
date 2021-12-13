import os
import torch
import hydra
import logging
import pandas as pd

from typing import Tuple
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, DataLoader


log = logging.getLogger(__name__)


class DDoSDataset(Dataset):
    """
    Torch based dataset ready for dataloader
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> None:
        log.debug("Initializing DDoSDataset")

        self.X = X.to(device) if X.device != device else X
        self.y = y.to(device) if y.device != device else y

        log.debug("DDoSDataset initialized")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloader(
        path: str,
        batch_size: int,
        num_workers: int=0,
        device: torch.device=torch.device('cpu'),
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates the dataloaders for the training, validation and test set.

    :param path: The path to where the .pt files are located.
    :param batch_size: The batch size of the dataloaders.
    :param num_workers: The number of workers to use for the dataloaders.
    :param device: The device to use for the dataloaders.
    :param kwargs: The kwargs to pass to create_sets.

    :return: Training, validation and test dataloaders.
    """

    log.debug(f'Creating the dataloaders for the training, validation and test set.')

    X, y = torch.empty(0, device=device, dtype=torch.float), torch.empty(0, device=device, dtype=torch.long)

    for f in os.listdir(path):
        if f.endswith(".pt"):
            log.debug(f'Loading {os.path.join(path, f)}')

            new_X, new_y = torch.load(os.path.join(path, f), map_location=device)

            X = torch.cat((X, new_X), dim=0)
            y = torch.cat((y, new_y), dim=0)

    train_ration = kwargs.get('train_ration', 0.8)
    validation_ration = kwargs.get('validation_ration', 0.1)

    (X_train, y_train), (X_validation, y_validation), (X_test, y_test) = create_sets(X, y, train_ration, validation_ration)

    train_loader = DataLoader(DDoSDataset(X_train, y_train, device), batch_size=batch_size, num_workers=num_workers)
    validation_loader = DataLoader(DDoSDataset(X_validation, y_validation, device), batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(DDoSDataset(X_test, y_test, device), batch_size=batch_size, num_workers=num_workers)

    log.info(f'Dataloaders created')

    return train_loader, validation_loader, test_loader


def dataframe_to_tensor(df: pd.DataFrame, device: str='cpu', dtype: torch.dtype=torch.float) -> torch.Tensor:
    """
    Converts a pandas dataframe to a tensor.

    :param df: The dataframe to convert.
    :param device: The device to convert the dataframe to.
    :param dtype: The dtype to convert the dataframe to.

    :return: The tensor containing the dataframe data.
    """

    return torch.tensor(df.values, device=device, dtype=dtype)


def create_sequences(data: torch.Tensor, labels: torch.Tensor, sequence_length: int=1, step: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates sequences from a tensor.

    :param data: The tensor to create sequences from.
    :param labels: The labels of the data.
    :param sequence_length: The length of the sequences.
    :param step: The step size of the sequences.

    :return: Tuple containing the data and labels sequenced.
    """

    return data.unfold(0, sequence_length, step).permute(0, 2, 1), labels.unfold(0, sequence_length, step)


def get_label_of_sequence(data: torch.Tensor) -> torch.Tensor:
    """
    Get the label of a sequence based on its last element.

    :param data: The data to get the labels of.

    :return: The labels of the data.
    """

    assert len(data.shape) == 2, f'The sequence must be a 2D tensor (-1, seq_size) but received a {len(data.shape)}D tensor.'

    labels = []

    for seq in data:
        labels.append(seq[-1].item())

    return torch.tensor(labels, device=data.device)


def get_occurences_of_labels(df: pd.DataFrame, label_column: str) -> Tuple[int, int]:
    """
    Returns the number of occurences of each label.

    :param df: the dataframe to be checked
    :param label_column: the column of the labels

    :return: the number of occurences of each label
    """

    return len(df[df[label_column] == 0]), len(df[df[label_column] == 1])


def limit_categories_in_tensors(data: torch.Tensor, labels: torch.Tensor, max_occurence) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Limits the number of categories in the data and labels.

    :param data: the data to be checked
    :param labels: the labels to be checked
    :param max_occurence: the maximum number of occurence of each label

    :return: the data and labels with the limited number of categories
    """

    log.debug(f'Limiting the number of categories in the data and labels.')

    selected_data = []
    selected_labels = []
    occurences = {0: 0, 1: 0}

    for data, label in zip(data, labels):
        label_value = label.item()

        if occurences[label_value] >= max_occurence:
            continue

        occurences[label_value] += 1


        selected_data.append(data)
        selected_labels.append(label)

    log.info(f'Limited the number of categories in the data and labels.')

    return torch.stack(selected_data), torch.stack(selected_labels)


def shuffle_set(data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffles a set.

    :param data: The data to shuffle.
    :param labels: The labels to shuffle.

    :return: The shuffled data and labels.
    """

    permutation = torch.randperm(data.shape[0])

    return data[permutation], labels[permutation]


def create_sets(data: torch.Tensor, labels: torch.Tensor, train_ratio: int=0.8, validation_ratio: int=0.1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates the training, validation and test sets from a tensor.

    :param data: The tensor to create the sets from.
    :param labels: The labels of the data.
    :param train_size: The size of the training set.
    :param validation_size: The size of the validation set.

    :return: Tuple containing the training, validation and test sets.
    """

    train_size = int(train_ratio * data.shape[0])
    validation_size = int(validation_ratio * data.shape[0])

    X_train, y_train = data[:train_size], labels[:train_size]
    X_validation, y_validation = data[train_size:train_size + validation_size], labels[train_size:train_size + validation_size]
    X_test, y_test = data[train_size + validation_size:], labels[train_size + validation_size:]

    return (X_train, y_train), (X_validation, y_validation), (X_test, y_test)


def save_tensor(tensor: torch.Tensor, file_path: str) -> None:
    """
    Saves a tensor to a file.

    :param tensor: The tensor to save.
    :param file_path: The path to save the tensor to.
    """

    torch.save(tensor, file_path)


def __create_tensors_from_df(df: pd.DataFrame, sequence_size: int, step: int, device: str='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates tensors from a clened dataframe.

    :param df: The dataframe to create the tensor from.
    :param sequence_size: The sequence size to use
    :param device: The device to create the tensor to.

    :return: The tensors created from the dataframe.
    """

    log.debug(f'Creating tensors from dataframe')

    with torch.no_grad():

        max_occurence = min(get_occurences_of_labels(df, 'Label')) // sequence_size

        X = dataframe_to_tensor(df.drop('Label', axis=1), device=device)
        y = dataframe_to_tensor(df['Label'], device=device, dtype=torch.long)

        del df

        X, y = create_sequences(X, y, sequence_length=sequence_size, step=step)
        y = get_label_of_sequence(y)


        X, y = shuffle_set(X, y)

        X, y = limit_categories_in_tensors(X, y, max_occurence=max_occurence)
        X, y = shuffle_set(X, y)

        log.info(f'Tensors created from dataframe')

        return X, y


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Create tensors files from the normalized dataframe.

    :param cfg: The configuration to use.
    """

    data_cfg = cfg.data
    ressource_cfg = cfg.ressource

    path = to_absolute_path(data_cfg["data_config"]["n-merge_path"])
    chuncksize = ressource_cfg["chunksize"]

    with pd.read_csv(path, chunksize=chuncksize) as reader:
        idx = 0
        X, y = torch.empty(0, dtype=torch.float), torch.empty(0, dtype=torch.long)

        for i, chunk in enumerate(reader):

            log.debug(f'Creating tensor for chunk {i}')

            new_X, new_y = __create_tensors_from_df(
                chunk,
                sequence_size=data_cfg["tensors"]["sequence_size"],
                step=data_cfg["tensors"]["step"],
                device=torch.device(ressource_cfg['device'])
            )

            X = torch.cat((X, new_X), dim=0)
            y = torch.cat((y, new_y), dim=0)

            if len(X) >= ressource_cfg['max_tensor_size']:
                save_tensor((X, y), to_absolute_path(eval(data_cfg["tensors"]["path"])))

                X, y = torch.empty(0), torch.empty(0)
                idx += 1

                log.info(f'Tensor for chunk {i} saved')


        if len(X) > 0:
            save_tensor((X, y), to_absolute_path(eval(data_cfg["tensors"]["path"])))

            log.info(f'Tensor for last chunk saved')


if __name__ == '__main__':
    main()
