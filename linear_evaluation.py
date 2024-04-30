import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from dataset import CustomCocoDetection, CustomCocoClassification

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils import yaml_config_hook


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device, val_loader=None):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    if val_loader is not None:
        val_X, val_y = inference(val_loader, simclr_model, device)
        return train_X, train_y, test_X, test_y, val_X, val_y
    else:
        return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size, X_val=None, y_val=None):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    if X_val is not None and y_val is not None:
        val = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer, save_confusion=False):
    loss_epoch = 0
    accuracy_epoch = 0
    true_labels = []
    predicted_labels = []
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        true_labels.extend(y.numpy())
        predicted_labels.extend(predicted.numpy())
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )
    if save_confusion:
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot confusion matrix
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('train_confusion_matrix.png')
        plt.close()

    return loss_epoch, accuracy_epoch

def validate(args, loader, simclr_model, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    true_labels = []
    predicted_labels = []
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        true_labels.extend(y.numpy())
        predicted_labels.extend(predicted.numpy())

        loss_epoch += loss.item()

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('test_confusion_matrix.png')
    plt.close()

    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "FDD":
        train_dir = os.path.join(args.dataset_dir, "train")
        test_dir = os.path.join(args.dataset_dir, "test")
        val_dir = os.path.join(args.dataset_dir, "val")
        train_dataset = CustomCocoClassification(train_dir, 
                                            transform=TransformsSimCLR(size=args.image_size).test_transform)
        test_dataset = CustomCocoClassification(test_dir,
                                            transform=TransformsSimCLR(size=args.image_size).test_transform)
        val_dataset = CustomCocoClassification(val_dir, 
                                               transform=TransformsSimCLR(size=args.image_size).test_transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    print("n_features: ", n_features)

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    print("simclr n_features: ", simclr_model.n_features)
    n_classes = 40 #10  # CIFAR-10 / STL-10
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y, val_X, val_y) = get_features(
        simclr_model, train_loader, test_loader, args.device, val_loader
    )

    arr_train_loader, arr_test_loader, arr_val_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size, val_X, val_y
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(args.logistic_epochs):
        save_confusion = (epoch == args.logistic_epochs - 1)
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, simclr_model, model, criterion, optimizer, save_confusion=save_confusion
        )
        val_loss, val_acc = validate(args, arr_val_loader, simclr_model, model, criterion)
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )
        train_losses.append(loss_epoch / len(arr_train_loader))
        train_accuracies.append(accuracy_epoch / len(arr_train_loader))
        val_losses.append(val_loss/len(arr_val_loader))
        val_accuracies.append(val_acc / len(arr_val_loader))

    # final tests
    loss_epoch, accuracy_epoch = test(
        args, arr_test_loader, simclr_model, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )

    print("train_losses: ", train_losses)
    print("train_accuracies: ", train_accuracies)
    print("validation_losses: ", val_losses)
    print("validation_accuracies: ", val_accuracies)

    epochs = range(1, len(train_losses) + 1)

    # Plot train loss
    plt.plot(epochs, train_losses, label='Train Loss')

    # Plot validation loss
    plt.plot(epochs, val_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.grid(True)

    # Add legend
    plt.legend()

    # Save the figure
    plt.savefig('train_losses_plot.png')