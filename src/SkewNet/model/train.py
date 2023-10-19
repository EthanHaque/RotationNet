import torch
from tqdm import tqdm


def circular_mse(y_pred, y_true):
    """Compute the circular mean squared error between two tensors.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predictions from the model. Expected to be a tensor of shape
        (batch_size, 1).
    y_true : torch.Tensor
        The ground truth values. Expected to be a tensor of shape
        (batch_size, 1).

    Returns
    -------
    torch.Tensor
        The circular mean squared error between the two tensors.
    """
    error = torch.atan2(torch.sin(y_pred - y_true), torch.cos(y_pred - y_true))
    return torch.mean(error ** 2)



def train(model, train_loder, optimizer, devie, *args):
    model.train()

    total_loss = 0.0
    for data, target in enumerate(tqdm(train_loder, desc="Training", leave=False)):
        data, target = data.to(devie), target.to(devie)

        optimizer.zero_grad()
        output = model(data)
        loss = circular_mse(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loder)
    # TODO remove print statement
    print(f"Train loss: {average_loss}")
    return average_loss


def test(model, test_loader, device, *args):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = circular_mse(output, target)

            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    # TODO remove print statement
    print(f"Test loss: {average_loss}")
    return average_loss


def validation(model, validation_loader, device, *args):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(validation_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = circular_mse(output, target)

            total_loss += loss.item()

    average_loss = total_loss / len(validation_loader)
    # TODO remove print statement
    print(f"Validation loss: {average_loss}")
    return average_loss
    