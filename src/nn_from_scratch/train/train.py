import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader, TensorDataset
from .preprocessing import data_split_and_preprocess
from .models import HeartDiseaseDetector
import os
import json
from typing import Sequence


def validate(data_loader: DataLoader, model: nn.Module, device: str | torch.device = "cuda", threshold: float = 0.5):
    model.eval()
    loss_fn = nn.BCELoss(reduction='sum')
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X.to(device))
            loss += loss_fn(output, y.to(device)).detach().item()
            correct += ((output >= threshold).float() == y.to(device)).sum().detach().item()
            total += y.size(0)
    return loss / total, correct / total


def train_loop(train_dataloader: DataLoader,
               valid_dataloader: DataLoader,
               optimizer: optim.Optimizer,
               loss_fn: nn.Module,
               epochs: int,
               model: nn.Module,
               save_path: str,
               device: str | torch.device | None = None,
               global_acc: float = 0.,
               log_step: int = 1000) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_acc = global_acc
    for epoch in range(1, epochs + 1):
        model.train()
        for X_train, y_train in train_dataloader:
            outputs = model(X_train.to(device))
            loss = loss_fn(outputs, y_train.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = validate(train_dataloader, model, device=device)
        valid_loss, valid_acc = validate(valid_dataloader, model, device=device)

        if valid_acc > max_acc:
            max_acc = valid_acc
            torch.save(model.state_dict(), save_path)

        if epoch == 1 or epoch % log_step == 0:
            print('{} Epoch {}'.format(datetime.datetime.now(), epoch))
            print("Training loss {:.5f} Validation loss {:.5f}".format(train_loss, valid_loss))
            print("Training Accuracy: {:.5f} Validation Accuracy: {:.5f}".format(train_acc, valid_acc))
            print()
    return max_acc


def train(data_path: str,
          model_path: str,
          epochs: int = 4000,
          device: str | torch.device | None = None,
          train_batch_size: int = 512,
          val_batch_size: int = 512,
          hidden_size_variants: Sequence[Sequence[int]] = ((), (16,), (32,), (32, 16)),
          use_batch_norm_variants: Sequence[bool] = (False, True),
          use_dropout_variants: Sequence[bool] = (False, True),
          dropout_rate_variants: Sequence[float] = (0.0, 0.2, 0.5),
          lr_variants: Sequence[float] = (0.1, 0.01)):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), preprocessing = data_split_and_preprocess(data_path=data_path, model_path=model_path)

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"y_valid: {y_valid.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    input_size = X_train.shape[1]
    print(f"Input_size: {input_size}")

    X_train_tensor = torch.from_numpy(X_train).to(torch.float32)
    X_valid_tensor = torch.from_numpy(X_valid).to(torch.float32)
    X_test_tensor = torch.from_numpy(X_test).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    y_valid_tensor = torch.from_numpy(y_valid).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

    train_dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=train_batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor), batch_size=val_batch_size,
                                  shuffle=False)
    test_dataloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=val_batch_size,
                                 shuffle=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_acc = 0.0
    for hidden_sizes in hidden_size_variants:
        for use_batch_norm in use_batch_norm_variants:
            for use_dropout in use_dropout_variants:
                for dropout_rate in dropout_rate_variants:
                    for lr in lr_variants:
                        if not use_dropout and dropout_rate > 0.0:
                            break

                        if len(hidden_sizes) == 0 and (use_batch_norm or use_dropout):
                            break

                        params = {
                            "hidden_sizes": hidden_sizes,
                            "use_batch_norm": use_batch_norm,
                            "dropout_rate": dropout_rate,
                            "use_dropout": use_dropout,
                            "lr": lr,
                            "input_size": input_size
                        }
                        print(params)
                        model = HeartDiseaseDetector(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1,
                                                     use_batch_norm=use_batch_norm,
                                                     dropout_rate=dropout_rate, use_dropout=use_dropout).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        max_acc_on_val = train_loop(
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            optimizer=optimizer,
                            loss_fn=nn.BCELoss(),
                            epochs=epochs,
                            model=model,
                            save_path=os.path.join(model_path, "heart_disease_detector.pt"),
                            device=device,
                            global_acc=max_acc,
                        )
                        if max_acc_on_val > max_acc:
                            max_acc = max_acc_on_val
                            with open(os.path.join(model_path, "nn_parameters.json"), "w") as f:
                                json.dump(params, f)

                        print()
                        print("Max val acc: {}".format(max_acc))
                        print("-----------------------------------------------------")

    with open(os.path.join(model_path, "nn_parameters.json"), "r") as f:
        training_params = json.load(f)

    if not isinstance(training_params['hidden_sizes'], Sequence):
        raise Exception("hidden must be sequence of integers")

    if type(training_params['dropout_rate']) != float:
        raise Exception("Dropout rate should be a float value")

    if type(training_params['use_dropout']) != bool:
        raise Exception("Use dropout should be a boolean value")

    if type(training_params['use_batch_norm']) != bool:
        raise Exception("Use Batch Normalization should be a boolean value")

    model = (HeartDiseaseDetector(input_size=input_size,
                                  hidden_sizes=training_params['hidden_sizes'],
                                  output_size=1,
                                  use_batch_norm=training_params['use_batch_norm'],
                                  use_dropout=training_params['use_dropout'],
                                  dropout_rate=training_params['dropout_rate']).to(device))

    model.load_state_dict(torch.load(os.path.join(model_path, "heart_disease_detector.pt"), map_location=device, weights_only=True))

    test_loss, test_acc = validate(test_dataloader, model, device=device)
    val_loss, val_acc = validate(valid_dataloader, model, device=device)
    train_loss, train_acc = validate(train_dataloader, model, device=device)

    print(" Train accuracy: {:.5f}".format(train_acc))
    print(" Valid accuracy: {:.5f}".format(val_acc))
    print(" Test accuracy: {:.5f}".format(test_acc))
