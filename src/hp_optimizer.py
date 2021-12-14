import LSTM
import hydra
import torch
import mlflow

from omegaconf import DictConfig
from Dataset import create_dataloader
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig


def log_omegaconf_params_to_mlflow(hopt_cfg) -> None:
    """
    Logs the hyperparameters used to train the model to MLflow.

    :param hopt_cfg: The hyperparameter optimization configuration.

    :return: None
    """

    for key in hopt_cfg.search_space.keys():
        mlflow.log_param(key, hopt_cfg[key.split('.')[1]])


class __CallBackMlFlow(object):
    """
    Callback to log the metrics to MLflow.

    :attr:`epoch`: The current epoch.
    :attr:`step`: The current step.
    """

    def __init__(self, **_):
        """
        Initializes the callback.

        :param kwargs: The keyword arguments. (unused)
        """

        self.epoch = 0
        self.step = 0

    def __call__(self, **kwargs) -> None:
        """
        To call at the end of a step. Logs the metrics to MLflow.

        :param kwargs: The keyword arguments.

        :return: None
        """

        self.step += 1

        mlflow.log_metric('loss', kwargs['loss'].item(), step=self.step)

    def on_epoch_end(self, **_) -> None:
        """
        Called at the end of an epoch.

        :param _: The keyword arguments. (unused)

        :return: None
        """

        self.epoch += 1



@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Finds the best hyperparameters for the LSTM model.

    :param cfg: The configuration for the LSTM model.

    :return: The validation loss and the test loss added together.
    """

    data_cfg = cfg.data
    ressource_cfg = cfg.ressource
    hopt_cfg = cfg.hopt

    DEVICE = torch.device(ressource_cfg["device"])

    train_loader, val_loader, test_loader = create_dataloader(
        to_absolute_path(data_cfg["tensors"]["dir"]),
        batch_size=ressource_cfg["batch_size"],
        num_workers=ressource_cfg["num_workers"],
        device=torch.device(DEVICE),
        train_ratio=data_cfg["data_config"]["train_ratio"],
        validation_ratio=data_cfg["data_config"]["validation_ratio"],
    )

    input_size = len(data_cfg["input_data"][list(data_cfg["input_data"].keys())[0]]['columns']) - 1

    model = LSTM.LSTM(
        input_size=input_size,
        output_size=data_cfg["data_config"]["num_classes"],

        lstm_num_layers=hopt_cfg['values']['lstm_num_layers'],
        lstm_hidden_size=hopt_cfg['values']['lstm_hidden_size'],

        num_layers=hopt_cfg['values']['num_layers'],
        hidden_size=hopt_cfg['values']['hidden_size'],

        device=DEVICE,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, hopt_cfg['values']['optimizer_name'])(model.parameters(), lr=hopt_cfg['values']['lr'])

    mlflow.set_tracking_uri(to_absolute_path('mlruns'))
    mlflow.set_experiment("lstm-hopt")

    callback = __CallBackMlFlow()

    with mlflow.start_run(run_name=f'{hopt_cfg.sweeper.study_name}_{HydraConfig.get().job.num}', nested=True):
        mlflow.set_tag('hopt_lib', hopt_cfg.name)
        log_omegaconf_params_to_mlflow(hopt_cfg)

        for epoch in range(hopt_cfg.sweeper.epochs):
            LSTM.train_model(model, train_loader, criterion, optimizer, callbacks=[callback])

        mlflow.pytorch.log_model(model, "model")

        val_loss, _ = LSTM.test_model(model, val_loader, criterion)
        test_loss, _ = LSTM.test_model(model, test_loader, criterion)

        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)

    return val_loss + test_loss


if __name__ == "__main__":
    main()
