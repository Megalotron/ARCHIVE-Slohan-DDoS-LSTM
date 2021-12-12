import hydra
import logging
import pandas as pd

from omegaconf import DictConfig


log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Merge every preprocessed dataframe into one dataframe.

    :param cfg: The configuration to use.
    """

    data_cfg = cfg.data
    ressource_cfg = cfg.ressource

    columns = [column['rename'] for _, column in data_cfg["input_data"][list(data_cfg["input_data"].keys())[0]]['columns'].items()]
    pd.DataFrame(columns=columns).to_csv(to_absolute_path(data_cfg["data_config"]["merge_path"]), index=False)

    pbar = tqdm(data_cfg["input_data"].items(), total=len(data_cfg["input_data"]), leave=None, position=0)
    for dataset_name, sng_data_conf in pbar:

        pbar.set_description(f'Merging {dataset_name}')
        log.debug(f'Merging {dataset_name}')

        with pd.read_csv(to_absolute_path(sng_data_conf["preprocessed_path"]), chunksize=ressource_cfg["chunksize"]) as reader:
            for idx, chunk in enumerate(reader):
                log.debug(f'Merging chunk {idx}')

                chunk.to_csv(to_absolute_path(data_cfg["data_config"]["merge_path"]), index=False, header=False, mode='a')

                log.info(f'Finished merging chunk n°{idx}')

        log.info(f'Finished merging {dataset_name}')

    log.info(f'Finished merging all datasets')


if __name__ == "__main__":
    from tqdm import tqdm
    from omegaconf import DictConfig
    from hydra.utils import to_absolute_path

    main()
