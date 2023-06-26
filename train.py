import argparse
import os
import json
import random
import shutil
from collections import defaultdict

from torch import softmax
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs import model_config, path_config
from src.constants import resnet_type, feature_type, softmax_type, data_type, padding_type
from src.datasets.ASVSpoof_2019_dataset import ASVSpoof2019
from src.metrics.eval_metrics import compute_eer
from src.modules.oc_softmax import OCSoftmax
from src.models.resnet import ResNet

torch.set_default_tensor_type(torch.FloatTensor)


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    # data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default="LA")
    parser.add_argument("-f", "--path_to_features", type=str, help="features path", default=path_config.FEATURE_PATH)
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path", default=path_config.PROTOCOL_PATH)
    parser.add_argument("-o", "--out_folder", type=str, help="output folder", default="./models/try/")

    # dataset prepare
    parser.add_argument("--feat_len", type=int, help="features length", default=model_config.FEAT_LEN)
    parser.add_argument(
        "--padding",
        type=str,
        default=padding_type.REPEAT,
        choices=[padding_type.ZERO, padding_type.REPEAT],
        help="how to pad short utterance"
    )
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=model_config.NUM_EPOCHS, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=model_config.BATCH_SIZE, help="mini batch size for training")
    parser.add_argument("--lr", type=float, default=model_config.LEARNING_RATE, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=model_config.DECAY_LEARNING_RATE, help="decay learning rate")
    parser.add_argument("--interval", type=int, default=10, help="interval to decay lr")

    parser.add_argument("--beta_1", type=float, default=0.9, help="beta_1 for Adam")
    parser.add_argument("--beta_2", type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument("--seed", type=int, help="random number seed", default=598)

    parser.add_argument(
        "--add_loss",
        type=str,
        default=softmax_type.OC_SOFTMAX,
        choices=[softmax_type.NORMAL, softmax_type.OC_SOFTMAX],
        help="loss_functions for one-class training"
    )
    parser.add_argument("--weight_loss", type=float, default=1, help="weight for other loss_functions")
    parser.add_argument("--r_real", type=float, default=0.9, help="r_real for oc_softmax")
    parser.add_argument("--r_fake", type=float, default=0.2, help="r_fake for oc_softmax")
    parser.add_argument("--alpha", type=float, default=20, help="scale factor for oc_softmax")

    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="continue training with previously trained model"
    )
    args = parser.parse_args()

    # change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.continue_training:
        assert os.path.exists(args.out_folder)
    else:
        # path for output data
        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)
        else:
            shutil.rmtree(args.out_folder)
            os.mkdir(args.out_folder)

        # folder for intermediate results
        if not os.path.exists(os.path.join(args.out_folder, "checkpoint")):
            os.makedirs(os.path.join(args.out_folder, "checkpoint"))
        else:
            shutil.rmtree(os.path.join(args.out_folder, "checkpoint"))
            os.mkdir(os.path.join(args.out_folder, "checkpoint"))

        # path for input data
        assert os.path.exists(args.path_to_features)

        # save training args
        with open(os.path.join(args.out_folder, "args.json"), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_folder, "train_loss.log"), 'w') as file:
            file.write("Start recording training loss_functions ...\n")

        with open(os.path.join(args.out_folder, "dev_loss.log"), 'w') as file:
            file.write("Start recording validation loss_functions ...\n")

    # assign device
    args.cuda = torch.cuda.is_available()
    print("Cuda device available: ", args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def setup_seed(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    model = ResNet(
        enc_dim=args.enc_dim,
        model_type=resnet_type.TYPE_18_LAYERS,
        num_classes=2
    ).to(args.device)

    if args.continue_training:
        model = torch.load(os.path.join(args.out_folder, "anti-spoofing_lfcc_model.pt")).to(args.device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2), 
        eps=args.eps, 
        weight_decay=0.0005
    )

    training_set = ASVSpoof2019(
        access_type=args.access_type,
        path_to_features=args.path_to_features,
        path_to_protocol=args.path_to_protocol,
        part=data_type.TRAIN,
        feature_type=feature_type.DSCC,
        feat_len=args.feat_len, 
        padding_type=args.padding
    )
    validation_set = ASVSpoof2019(
        access_type=args.access_type,
        path_to_features=args.path_to_features,
        path_to_protocol=args.path_to_protocol,
        part=data_type.DEV,
        feature_type=feature_type.DSCC,
        feat_len=args.feat_len, 
        padding_type=args.padding
    )
    train_data_loader = DataLoader(
        dataset=training_set,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=training_set.collate_fn
    )
    val_data_loader = DataLoader(
        dataset=validation_set,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=validation_set.collate_fn
    )

    feat, _, _, _ = training_set[29]
    print("Feature shape", feat.shape)

    criterion = CrossEntropyLoss()
    softmax_loss = OCSoftmax(
        feat_dim=args.enc_dim,
        r_real=args.r_real,
        r_fake=args.r_fake,
        alpha=args.alpha
    ).to(args.device)
    softmax_loss.train()
    softmax_optimizer = torch.optim.SGD(softmax_loss.parameters(), lr=args.lr)

    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        model.train()
        train_loss_dict = defaultdict(list)
        dev_loss_dict = defaultdict(list)
        adjust_learning_rate(args, optimizer, epoch_num)
        adjust_learning_rate(args, softmax_optimizer, epoch_num)
        print("\nEpoch: %d " % (epoch_num + 1))
        for i, (lfcc, audio_fns, tags, labels) in enumerate(tqdm(train_data_loader)):
            lfcc = lfcc.unsqueeze(1).float().to(args.device)
            labels = labels.to(args.device)
            feats, lfcc_outputs = model(lfcc)
            lfcc_loss = criterion(lfcc_outputs, labels)

            if args.add_loss == softmax_type.NORMAL:
                optimizer.zero_grad()
                train_loss_dict[args.add_loss].append(lfcc_loss.item())
                lfcc_loss.backward()
                optimizer.step()

            if args.add_loss == softmax_type.OC_SOFTMAX:
                oc_softmax_loss, _ = softmax_loss(feats, labels)
                lfcc_loss = oc_softmax_loss * args.weight_loss
                optimizer.zero_grad()
                softmax_optimizer.zero_grad()
                train_loss_dict[args.add_loss].append(oc_softmax_loss.item())
                lfcc_loss.backward()
                optimizer.step()
                softmax_optimizer.step()

            with open(os.path.join(args.out_folder, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" + str(np.nanmean(train_loss_dict[monitor_loss])) + "\n")

        # Val the model
        model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(val_data_loader)):
                lfcc = lfcc.unsqueeze(1).float().to(args.device)
                labels = labels.to(args.device)

                feats, lfcc_outputs = model(lfcc)

                lfcc_loss = criterion(lfcc_outputs, labels)
                score = softmax(lfcc_outputs, dim=1)[:, 0]

                if args.add_loss == softmax_type.NORMAL:
                    dev_loss_dict[softmax_type.NORMAL].append(lfcc_loss.item())
                elif args.add_loss == softmax_type.OC_SOFTMAX:
                    oc_softmax_loss, score = softmax_loss(feats, labels)
                    dev_loss_dict[args.add_loss].append(oc_softmax_loss.item())
                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_folder, "dev_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" + str(np.nanmean(dev_loss_dict[monitor_loss])) + "\t" + str(val_eer) + "\n"
                )
            print("Val EER: {}".format(val_eer))

        torch.save(
            model,
            os.path.join(
                args.out_folder,
                "checkpoint",
                "anti-spoofing_lfcc_model_%d.pt" % (epoch_num + 1)
            )
        )
        loss_model = None
        if args.add_loss in [softmax_type.OC_SOFTMAX]:
            loss_model = softmax_loss
            torch.save(
                loss_model,
                os.path.join(
                    args.out_folder,
                    "checkpoint",
                    "anti-spoofing_loss_model_%d.pt" % (epoch_num + 1)
                )
            )

        if val_eer < prev_eer:
            torch.save(model, os.path.join(args.out_folder, 'anti-spoofing_lfcc_model.pt'))
            torch.save(loss_model, os.path.join(args.out_folder, 'anti-spoofing_loss_model.pt'))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_folder, "args.json"), 'a') as res_file:
                res_file.write("\nTrained Epochs: %d\n" % (epoch_num - 19))
            break

    return model, loss_model


def main():
    args = init_params()
    setup_seed(args.seed)
    _, _ = train(args)
    model = torch.load(os.path.join(args.out_folder, "anti-spoofing_lfcc_model.pt"))
    if args.add_loss == softmax_type.NORMAL:
        loss_model = None
    else:
        loss_model = torch.load(os.path.join(args.out_folder, "anti-spoofing_loss_model.pt"))


if __name__ == "__main__":
    main()
