# training a discrete normalizing flow
import yaml
import time
import os
import sys
import numpy as np
import torch
from device_utils import get_device
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from modded_basic_nflows import (
    create_mixture_flow_model,
    save_model,
    load_mixture_model,
)

from validation import validate
from src.validation import validate as validateBig
from data_preprocessing import (
    TrainDataPreprocessor,
    TestDataPreprocessor,
    TrainDataPreprocessorBig,
    TestDataPreprocessorBig,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal


def train(
    input_dim, context_dim, gpu, train_kwargs, data_kwargs, base_kwargs, validate_at_0
):
    device = get_device(gpu_requested=(gpu is not None))

    if train_kwargs["log_name"] is not None:
        log_dir = "./logs/%s" % train_kwargs["log_name"]
        save_dir = "./checkpoints/%s" % train_kwargs["log_name"]

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    else:
        log_dir = "./logs/time-%d" % time.time()
        save_dir = "./checkpoints/time-%d" % time.time()

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    model = create_mixture_flow_model(input_dim, context_dim, base_kwargs)
    # print total params number and stuff
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    # add to tensorboard
    writer.add_scalar("total_params", total_params, 0)

    lr = train_kwargs["lr"]

    start_epoch = 0
    epochs = train_kwargs["epochs"]
    batch_size = data_kwargs["batch_size"]
    if train_kwargs["resume_checkpoint"] is None and os.path.exists(
        os.path.join(save_dir, "checkpoint-latest.pt")
    ):
        resume_checkpoint = os.path.join(
            save_dir, "checkpoint-latest.pt"
        )  # use the latest checkpoint
    else:
        resume_checkpoint = train_kwargs["resume_checkpoint"]
    if resume_checkpoint is not None and train_kwargs["resume"] == True:
        model, _, lr, start_epoch, _, _ = load_mixture_model(
            model,
            model_dir=save_dir,
            filename="checkpoint-latest.pt",
        )
        print(f"Resumed from: {start_epoch}")

    # send model to device
    # model = buildFlow() # overwrite
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    scheduler = None
    if train_kwargs["scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30, verbose=True
        )
    elif train_kwargs["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5, verbose=True
        )
    else:
        print("Scheduler not found, proceeding without it")
    if input_dim == 5:
        train_dataset = TrainDataPreprocessor(data_kwargs)
        X_train, Y_train = train_dataset.get_dataset()
        test_dataset = TestDataPreprocessor(
            data_kwargs,
            scaler_x=train_dataset.scaler_x,
            scaler_y=train_dataset.scaler_y,
        )
        X_test, Y_test = test_dataset.get_dataset()
    elif input_dim == 16:
        train_dataset = TrainDataPreprocessorBig(data_kwargs)
        X_train, Y_train = train_dataset.get_dataset()
        test_dataset = TestDataPreprocessorBig(
            data_kwargs,
            scaler_x=train_dataset.scaler_x,
            scaler_y=train_dataset.scaler_y,
        )
        X_test, Y_test = test_dataset.get_dataset()
    else:
        raise ValueError("Input dim not supported")

    # send data to device
    X_train = torch.tensor(X_train).float().to(device)
    Y_train = torch.tensor(Y_train).float().to(device)
    # test copies on cpu for eval
    X_test_cpu = X_test
    Y_test_cpu = Y_test

    X_test = torch.tensor(X_test).float().to(device)
    Y_test = torch.tensor(Y_test).float().to(device)

    if data_kwargs["standardize"] == True:
        X_test_cpu = test_dataset.scaler_x.inverse_transform(X_test_cpu)
        if data_kwargs["flavour_ohe"]:
            Y_test_cpu[:, 0:5] = test_dataset.scaler_y.inverse_transform(Y_test_cpu[:, 0:5])
        else:
            Y_test_cpu = test_dataset.scaler_y.inverse_transform(Y_test_cpu)

    if data_kwargs["flavour_ohe"]:
        # Back to the flavour: 0 = 0,1,2,3,21, 4 = 4, 5 = 5
        if Y_test_cpu.shape[1] > 6:
            tmp = Y_test_cpu[:, 5:]
            b = tmp[:, 5]
            c = tmp[:, 4]
            flavour = np.zeros(len(b))
            flavour[np.where(b == 1)] = 5
            flavour[np.where(c == 1)] = 4

        Y_test_cpu = np.hstack((Y_test_cpu[:, :4], flavour.reshape(-1, 1), Y_test_cpu[:, 6].reshape(-1, 1)))
        print("YY", Y_test_cpu.shape)


    # apply np.rint to the number of constituents
    X_test_cpu[:, 4] = np.round(X_test_cpu[:, 4])
    if input_dim == 16:
        X_test_cpu[:, 11] = np.round(X_test_cpu[:, 11])
        X_test_cpu[:, 12] = np.round(X_test_cpu[:, 12])
        X_test_cpu[:, 14] = np.round(X_test_cpu[:, 14])

    if validate_at_0:
        print("Starting sampling")
        model.eval()
        samples_list = []
        with torch.no_grad():
            with tqdm(
                total=len(X_test) // batch_size, desc="Sampling", dynamic_ncols=True, ascii=True
            ) as pbar:
                for i in range(0, len(X_test), batch_size):
                    Y_batch = Y_test[i : i + batch_size]

                    samples = model.sample(1, context=Y_batch)
                    samples_list.append(samples.detach().cpu().numpy())
                    pbar.update(1)

        samples = np.concatenate(samples_list, axis=0)
        samples = np.array(samples).reshape((-1, X_test.shape[1]))

        if data_kwargs["standardize"] == True:
            samples = test_dataset.scaler_x.inverse_transform(samples)


        # apply np.rint to the number of constituents
        samples[:, 4] = np.rint(samples[:, 4])
        if input_dim == 16:
            samples[:, 11] = np.rint(samples[:, 11])  # ncharged
            samples[:, 12] = np.rint(samples[:, 12])  # nneutral
            samples[:, 14] = np.rint(samples[:, 14])  # nSV

        # clip to physical values based on the boundaries of X_test_cpu
        for i in range(samples.shape[1]):
            samples[:, i] = np.clip(
                samples[:, i],
                X_test_cpu[:, i].min(),
                X_test_cpu[:, i].max(),
            )

        print("Starting evaluation")
        if input_dim == 5:
            validate(samples, X_test_cpu, Y_test_cpu, save_dir, start_epoch, writer)
        elif input_dim == 16:
            validateBig(samples, X_test_cpu, Y_test_cpu, save_dir, start_epoch, writer)
        else:
            raise ValueError("Input dim not supported")

    print("Start epoch: %d End epoch: %d" % (start_epoch, epochs))
    train_history = []
    test_history = []
    printout_freq = 50
    # with torch.autograd.set_detect_anomaly(True): # for debugging
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        # print lr with 8 digits precison
        for param_group in optimizer.param_groups:
            print_lr = param_group["lr"]
            print(f"Current lr is {print_lr:.8f}")

        train_loss = torch.tensor(0.0).to(device)
        train_log_p = torch.tensor(0.0).to(device)
        train_log_det = torch.tensor(0.0).to(device)
        # now manually loop over batches
        # use tqdm for progress bar
        total_batches = len(X_train) // batch_size
        if len(X_train) % batch_size != 0:
            total_batches += 1
        with tqdm(total=total_batches, desc="Training", dynamic_ncols=True) as pbar:
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i : i + batch_size]
                Y_batch = Y_train[i : i + batch_size]

                optimizer.zero_grad()
                # loss = -model.log_prob(inputs=X_batch, context=Y_batch).mean()
                log_p, log_det = model(inputs=X_batch, context=Y_batch)
                loss = -torch.mean(log_p + log_det)
                train_loss += loss.item()
                train_log_p += torch.mean(-log_p).item()
                train_log_det += torch.mean(-log_det).item()
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch Loss": loss.item()})

        train_loss /= total_batches
        train_log_p /= total_batches
        train_log_det /= total_batches

        writer.add_scalar("loss", train_loss, epoch)
        writer.add_scalar("log_p", train_log_p, epoch)
        writer.add_scalar("log_det", train_log_det, epoch)
        train_history.append([train_loss])
        test_history.append([0])
        if scheduler is not None:
            scheduler.step(train_loss)
        print("Epoch: %d Loss: %f" % (epoch, train_loss))

        with torch.no_grad():
            model.eval()
            test_loss = torch.tensor(0.0).to(device)
            test_log_p = torch.tensor(0.0).to(device)
            test_log_det = torch.tensor(0.0).to(device)

            for i in range(0, len(X_test), batch_size):
                X_batch = X_test[i : i + batch_size]
                Y_batch = Y_test[i : i + batch_size]

                log_p, log_det = model(inputs=X_batch, context=Y_batch)
                loss = -torch.mean(log_p + log_det)
                # loss = -model.log_prob(inputs=X_batch, context=Y_batch).mean()
                test_loss += loss.item()
                test_log_p += torch.mean(-log_p).item()
                test_log_det += torch.mean(-log_det).item()

            test_loss /= total_batches
            test_log_p /= total_batches
            test_log_det /= total_batches

            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_log_p", test_log_p, epoch)
            writer.add_scalar("test_log_det", test_log_det, epoch)

            test_history[-1] = [test_loss]
            print("Test Loss: %f" % (test_loss))

        # store losses in a csv file as well
        csv_file_path = os.path.join(save_dir, "losses.csv")

        # Check if file exists to decide between write and append mode
        mode = "a" if os.path.exists(csv_file_path) else "w"

        # save to csv as well
        with open(csv_file_path, mode) as f:
            if mode == "w":
                f.write(
                    "epoch,train_loss, train_log_p, train_log_det, test_loss, test_log_p, test_log_det\n"
                )

            f.write(
                f"{epoch},{train_loss},{train_log_p},{train_log_det},{test_loss},{test_log_p},{test_log_det}\n"
            )

        if epoch % train_kwargs["eval_freq"] == 0:
            print("Starting sampling")
            model.eval()
            samples_list = []
            with torch.no_grad():
                with tqdm(
                    total=len(X_test) // batch_size, desc="Sampling", dynamic_ncols=True
                ) as pbar:
                    for i in range(0, len(X_test), batch_size):
                        Y_batch = Y_test[i : i + batch_size]

                        samples = model.sample(1, context=Y_batch)
                        samples_list.append(samples.detach().cpu().numpy())
                        pbar.update(1)

            samples = np.concatenate(samples_list, axis=0)
            samples = np.array(samples).reshape((-1, X_test.shape[1]))

            if data_kwargs["standardize"] == True:
                samples = test_dataset.scaler_x.inverse_transform(samples)

            if data_kwargs["physics_scaling"] == True:
                samples, _ = test_dataset.invert_physics_scaling(samples, Y_test_cpu)

            # apply np.rint to the number of constituents
            samples[:, 4] = np.rint(samples[:, 4])
            if input_dim == 16:
                samples[:, 11] = np.rint(samples[:, 11])
                samples[:, 12] = np.rint(samples[:, 12])
                samples[:, 14] = np.rint(samples[:, 14])
            # clip to physical values based on the boundaries of X_test_cpu
            for i in range(samples.shape[1]):
                samples[:, i] = np.clip(
                    samples[:, i],
                    X_test_cpu[:, i].min(),
                    X_test_cpu[:, i].max(),
                )

            print("Starting evaluation")
            if input_dim == 5:
                validate(samples, X_test_cpu, Y_test_cpu, save_dir, epoch, writer)
            elif input_dim == 16:
                validateBig(samples, X_test_cpu, Y_test_cpu, save_dir, epoch, writer)
            else:
                raise ValueError("Input dim not supported")

        if epoch % train_kwargs["save_freq"] == 0:
            save_model(
                epoch,
                model,
                scheduler=scheduler,
                train_history=train_history,
                test_history=test_history,
                name="model",
                model_dir=save_dir,
                optimizer=optimizer,
            )
            print("Saved model")


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        config_path = "../configs/" + args[1]
    else:
        config_path = "../configs/dummy_train_config.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    input_dim = config["input_dim"]
    context_dim = config["context_dim"]
    gpu = config["gpu"]
    validate_at_0 = config["validate_at_0"]
    train_kwargs = config["train_kwargs"]
    data_kwargs = config["data_kwargs"]
    base_kwargs = config["base_kwargs"]
    train(
        input_dim,
        context_dim,
        gpu,
        train_kwargs,
        data_kwargs,
        base_kwargs,
        validate_at_0,
    )
