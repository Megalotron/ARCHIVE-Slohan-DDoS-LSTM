import os
import LSTM
import torch
import hydra
import logging

from omegaconf import DictConfig
from Dataset import create_dataloader
from utils import plot_mulitple_values
from hydra.utils import to_absolute_path


log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Train the model based on the configuration.

    :param cfg: The configuration to use.
    """

    data_cfg = cfg.data
    ressource_cfg = cfg.ressource
    model_cfg = cfg.model

    DEVICE = torch.device(ressource_cfg["device"])

    train_loader, validation_loader, test_loader = create_dataloader(
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

        lstm_num_layers=model_cfg["parameters"]["lstm_num_layers"],
        lstm_hidden_size=model_cfg["parameters"]["lstm_hidden_size"],

        num_layers=model_cfg["parameters"]["num_layers"],
        hidden_size=model_cfg["parameters"]["hidden_size"],

        device=DEVICE,
    )

    criterion = getattr(torch.nn, model_cfg["criterion"])()
    optimizer = getattr(torch.optim, model_cfg["optimizer"]["name"])(model.parameters(), lr=model_cfg["optimizer"]["learning_rate"])

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    best_model = {
        'state_dict': model.state_dict(),
        'loss': float('inf'),
        'accuracy': 0.0,
    }

    if not os.path.exists(model_cfg['saving']["checkpoint_dir"]):
        os.makedirs(model_cfg['saving']["checkpoint_dir"])

    for epoch in range(ressource_cfg['epochs']):
        train_loss, train_acc = LSTM.train_model(model, train_loader, criterion, optimizer)
        val_loss, val_acc = LSTM.test_model(model, validation_loader, criterion)

        if model_cfg['saving']["save_interval"] > 0 and epoch % model_cfg['saving']["save_interval"] == 0:
            torch.save(model.state_dict(), os.path.join(model_cfg['saving']["checkpoint_dir"], f"model_epoch_{epoch}.pt"))

        if val_loss < best_model['loss']:
            log.debug(f"New best model found with loss {val_loss}")

            best_model['state_dict'] = model.state_dict()
            best_model['loss'] = val_loss
            best_model['accuracy'] = val_acc

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        log.info(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")

    model.load_state_dict(best_model['state_dict'])

    test_loss, test_acc = LSTM.test_model(model, test_loader, criterion)
    log.info(f"Test loss: {test_loss:.3f} Test accuracy: {test_acc:.3f}")

    plot_mulitple_values([train_losses, val_losses], ['train loss', 'validation loss'], 'Loss', 'Epoch', 'Loss')
    plot_mulitple_values([train_accuracies, val_accuracies], ['train', 'validation'], 'Accuracy', 'Epoch', 'Accuracy')

    if not os.path.exists(os.path.dirname(model_cfg['saving']["best_model_path"])):
        os.makedirs(os.path.dirname(model_cfg['saving']["best_model_path"]))

    torch.save(best_model["state_dict"], model_cfg["saving"]["best_model_path"])


if __name__ == '__main__':
    main()
