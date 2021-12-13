import hydra
import logging
import pandas as pd

from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


log = logging.getLogger(__name__)


def drop_columns(df: pd.DataFrame, columns: list, inplace=False) -> pd.DataFrame:
    """
    Drops the specified columns from a dataframe.

    :param df: The dataframe to drop columns from.
    :param columns: The columns to drop.
    :param inplace: Whether to drop the columns inplace or return a new dataframe.

    :return: The dataframe with the specified columns dropped.
    """

    assert isinstance(df, pd.DataFrame), 'The dataframe must be a pandas dataframe.'
    assert isinstance(columns, list), 'The columns must be a list.'

    log.debug(f'Dropping columns {columns} from the dataframe.')

    if not inplace: return df.drop(columns, axis=1)

    df.drop(columns, axis=1, inplace=True)

    log.info(f'Dropped columns {columns} from the dataframe.')

    return df


def __preprocess_data_according_to_config(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocess the DataFrame according to its config.

    :param df: The DataFrame to be preprocessed.
    :param config: The config to use.

    :return: The preprocessed DataFrame.
    """

    log.debug(f'preprocessing data according to config')

    df.dropna(inplace=True)

    columns_to_drop = [col for col in df.columns if col not in config["columns"]]
    drop_columns(df, columns_to_drop, inplace=True)

    for column, col_cfg in config["columns"].items():

        log.debug(f'Prerocessing {col_cfg["rename"]}')

        df[column] = df[column].apply(eval(col_cfg["apply"]))

        if column != col_cfg["rename"]:
            df.rename(columns={column: col_cfg["rename"]}, inplace=True)

        log.info(f'{col_cfg["rename"]} preprocessed')

    log.info(f'Data preprocessed according to config')

    return df


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Execut every instructions (rename & apply) stored in the config.

    :param cfg: The configuration to use.
    """

    data_cfg = cfg.data
    ressource_cfg = cfg.ressource

    pbar = tqdm(data_cfg["input_data"].items(), total=len(data_cfg["input_data"]), leave=None, position=0)
    for dataset_name, sng_data_conf in pbar:

        pbar.set_description(f'Preprocessing {dataset_name}')
        log.debug(f'Preprocessing {dataset_name}')

        columns = [column['rename'] for _, column in sng_data_conf["columns"].items()]
        pd.DataFrame(columns=columns).to_csv(to_absolute_path(sng_data_conf["preprocessed_path"]), index=False)

        with pd.read_csv(to_absolute_path(sng_data_conf["raw_path"]), chunksize=ressource_cfg["chunksize"]) as reader:
            for idx, chunk in enumerate(reader):
                log.debug(f'Preprocessing chunk {idx}')

                chunk = __preprocess_data_according_to_config(chunk, sng_data_conf)
                chunk.to_csv(to_absolute_path(sng_data_conf["preprocessed_path"]), index=False, header=False, mode='a')

                log.info(f'Finished preprocessing chunk n°{idx}')

        log.info(f'Finished preprocessing {dataset_name}')

    log.info(f'Finished preprocessing all datasets')


if __name__ == '__main__':
    main()
