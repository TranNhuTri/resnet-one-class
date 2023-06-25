import argparse
import os
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.metrics import eval_metrics as em
from src.configs import model_config

from src.constants import softmax_type, feature_type
from src.datasets.ASVSpoof_2019_dataset import ASVSpoof2019


def test_model(feat_model_path, loss_model_path, part, add_loss, device):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location=torch.device(device))
    model = model.to(device)
    loss_model = torch.load(loss_model_path, map_location=torch.device(device)) if add_loss != "softmax" else None
    test_set = ASVSpoof2019(
        "LA",
        "./data/ASVSpoof2019/LA/features",
        "./data/ASVSpoof2019/LA/ASVSpoof2019_LA_cm_protocols/",
        part,
        feature_type=feature_type.LFCC,
        feat_len=model_config.FEAT_LEN,
        padding_type="repeat"
    )
    test_data_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=test_set.collate_fn
    )
    model.eval()

    with open(os.path.join(dir_path, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(test_data_loader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = softmax(lfcc_outputs)[:, 0]

            if add_loss == softmax_type.OC_SOFTMAX:
                ang_iso_loss, score = loss_model(feats, labels)
            elif add_loss == softmax_type.AM_SOFTMAX:
                outputs, m_outputs = loss_model(feats, labels)
                score = softmax(outputs, dim=1)[:, 0]

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    eer_cm, min_tDCF = em.compute_eer_and_tdcf(
        os.path.join(dir_path, 'checkpoint_cm_score.txt'),
        "data/ASVSpoof2019"
    )
    return eer_cm, min_tDCF


def test(model_dir, add_loss, device):
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model_2.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model_2.pt")
    test_model(model_path, loss_model_path, "eval", add_loss, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-m',
        '--model_dir',
        type=str,
        help="path to the trained model",
        default="./models/trained/oc_softmax"
    )
    parser.add_argument(
        '-l',
        '--loss',
        type=str,
        default=softmax_type.OC_SOFTMAX,
        choices=["softmax", 'am_softmax', 'oc_softmax'],
        help="loss_functions function"
    )
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.loss, args.device)
