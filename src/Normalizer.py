import json
import hydra
import logging
import pandas as pd

from torch import Tensor
from omegaconf import DictConfig
from multimethod import multimethod
from typing import List, Tuple, Dict


log = logging.getLogger(__name__)


class Normalizer:
    """
    This class is used to normalize the data.
    
    :attr:`params` is a dictionary containing the normalization parameters.
    """

    def __init__(self, path: str=None) -> None:
        """
        Initializes the normalizer.

        :param path: the path to the normalization parameters
        """

        self.params = None

        if path is not None:
            self.load_params(path)


    def load_params(self, filepath: str) -> None:
        """
        Loads the normalization parameters from a file.

        :param filepath: the filepath to the file containing the normalization parameters

        :return: None
        """

        self.params = json.load(open(filepath, 'r'))


    def save_params(self, filepath: str) -> None:
        """
        Saves the normalization parameters to a file.

        :param filepath: the filepath to the file to save the normalization parameters to
        :param key: the key to save the normalization parameters under

        :return: None
        """

        json.dump(self.params, open(filepath, 'w+'))


    @multimethod
    def __call__(self, *args):
        raise NotImplementedError(f"__call__ not implemented for type: {[type(arg) for arg in args]}")

    @multimethod
    def __call__(self, df: pd.DataFrame, inplace=False) -> pd.DataFrame:
        """
        Normalizes the dataframe.

        :param df: the data to be normalized
        :param params: the normalization parameters
        :param inplace: whether to normalize the dataframe inplace or not

        :return: the normalized dataframe
        """

        assert self.params is not None, "Normalization parameters are not set."

        if inplace:
            df_cpy = df
        else:
            df_cpy = df.copy()

        for column in df_cpy.columns:

            if column not in self.params:
                continue

            mean, std = self.params[column]['mean'], self.params[column]['std']

            df_cpy[column] = (df_cpy[column] - mean) / std

        return df_cpy

    @multimethod
    def __call__(self, sample: Dict[str, float]) -> Tuple[float, ...]:
        """
        Normalizes the sample.

        :param sample: the sample to be normalized

        :return: the normalized sample
        """

        for column in sample.keys():
            mean, std = self.params[column]['mean'], self.params[column]['std']

            sample[column] = (sample[column] - mean) / std

        return sample

    @multimethod
    def __call__(self, sample: List[float]) -> Tuple[float, ...]:
        """
        Normalizes the sample.

        :param sample: the sample to be normalized

        :return: the normalized sample
        """

        assert len(sample) == len(self.parms), f"data must be of lenght {len(self.params)} but received {len(sample)}"

        for idx, key in enumerate(self.params):
            mean, std = self.params[key]['mean'], self.params[key]['std']

            sample[idx] = (sample[idx] - mean) / std

        return sample

    @multimethod
    def __call__(self, sample: Tensor) -> Tensor:
        """
        Normalizes the sample.

        :param sample: the tensor to be normalized

        :return: the normalized tensor
        """

        assert len(sample.shape) == len(self.parms), f"tensor must be of shape ({len(self.params)}) but received {sample.shape}"

        for idx, key in enumerate(self.params):
            mean, std = self.params[key]['mean'], self.params[key]['std']

            sample[idx] = (sample[idx] - mean) / std

        return sample


def __get_csv_normalization_params(path: str, chuncksize: int) -> Dict[str, Dict[str, float]]:
    """
    Gets the normalization parameters for each chunk of a csv file.

    :param path: the path to the csv's
    :param chuncksize: the size of the chunks to read from the csv's

    :return: the normalization parameters
    """

    metadata = {}

    with pd.read_csv(path, chunksize=chuncksize) as reader:
        for idx, chunk in enumerate(reader):
            log.debug(f'Caclulating normalization params for chunk {idx}')

            for column in chunk.columns:

                if column != 'Label':

                    if column not in metadata:
                        metadata[column] = []

                    metadata[column].append({
                        'mean': chunk[column].mean(),
                        'std': chunk[column].std(),
                        'len': len(chunk[column])
                    })

            log.info(f'Finished caclulating normalization params for chunk n°{idx}')

    return metadata


def __get_normalization_params_from_metadata(metadata: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Gets the normalization parameters from the metadata.

    :param metadata: the metadata to get the normalization parameters from

    :return: the normalization parameters
    """

    log.debug(f'Getting normalization params from metadata')

    normalization_params = {}
    for column, params in metadata.items():
        nb_rows = sum([param['len'] for param in params])

        normalization_params[column] = {
            'mean': sum([param['mean'] * (param['len'] / nb_rows) for param in params]),
            'std': sum([param['std'] * (param['len'] / nb_rows) for param in params]),
        }

    log.info(f'Finished getting normalization params from metadata')

    return normalization_params


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Calculates the normalization parameters for the dataset, save it and save a normalized version of the dataset.

    :param cfg: the config to use

    :return: None
    """

    data_cfg = cfg.data
    ressource_cfg = cfg.ressource

    log.debug('Normalizing data')

    metadata = __get_csv_normalization_params(to_absolute_path(data_cfg["data_config"]["merge_path"]), ressource_cfg["chunksize"])
    normalization_params = __get_normalization_params_from_metadata(metadata)

    json.dump(normalization_params, open(to_absolute_path(data_cfg["data_config"]["normalization_params_path"]), 'w'))

    log.info(f'Normalization params saved')


    normalizer = Normalizer()
    normalizer.params = normalization_params

    columns = [column['rename'] for _, column in data_cfg["input_data"][list(data_cfg["input_data"].keys())[0]]['columns'].items()]
    pd.DataFrame(columns=columns).to_csv(to_absolute_path(data_cfg["data_config"]["n-merge_path"]), index=False)

    path = to_absolute_path(data_cfg["data_config"]["merge_path"])
    chunksize = ressource_cfg["chunksize"]

    with pd.read_csv(path, chunksize=chunksize) as reader:
        for idx, chunk in enumerate(reader):
            log.debug(f'Normalizing chunk {idx}')

            normalizer(chunk, inplace=True)

            chunk.to_csv(to_absolute_path(data_cfg["data_config"]["n-merge_path"]), mode='a', header=False, index=False)

            log.info(f'Finished normalizing chunk n°{idx}')

    log.info(f'Finished normalizing data')


if __name__ == "__main__":
    from omegaconf import DictConfig
    from hydra.utils import to_absolute_path

    main()
