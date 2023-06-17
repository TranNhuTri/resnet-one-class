import os
import pickle

import torchaudio
from torchaudio import transforms

from src.configs import lfcc_config
from src.constants import data_type, feature_type

features_path = "/"


def extract_features(
    part,
    feature,
    access_type="LA",
    proto_files_path="../data/ASVSpoof2019/LA/ASVSpoof2019_LA_cm_protocols",
    voice_files_path="/Volumes/T7/dataset/LA"
):
    if part == data_type.TRAIN:
        proto_file_extension = ".trn.txt"
    else:
        proto_file_extension = ".trl.txt"
    protocol_path = os.path.join(
        proto_files_path,
        "ASVSpoof2019." + access_type + ".cm." + part + proto_file_extension
    )
    with open(protocol_path, 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]
    for info in audio_info:
        speaker, filename, _, tag, label = info
        audio_path = os.path.join(
            voice_files_path,
            "ASVSpoof2019_" + access_type + "_" + part,
            "flac",
            filename + ".flac"
        )
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        win_length = int((sample_rate / 1000) * lfcc_config.WIN_LENGTH)
        if feature == feature_type.LFCC:
            transform = transforms.LFCC(
                sample_rate=sample_rate,
                n_lfcc=lfcc_config.N_LFCC,
                speckwargs={"n_fft": lfcc_config.N_FFT, "win_length": win_length},
            )
            features = transform(waveform)[0]
        else:
            features = None
        with open(
            os.path.join(
                features_path,
                part,
                feature,
                filename + '.pkl'
            ),
            "wb"
        ) as f:
            pickle.dump(features.numpy(), f)


if __name__ == "__main__":
    # extract_features(data_type.TRAIN, feature_type.LFCC)
    # extract_features(data_type.DEV, feature_type.LFCC)
    extract_features(data_type.EVAL, feature_type.LFCC)
