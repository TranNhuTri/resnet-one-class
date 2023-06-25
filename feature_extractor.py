import os
import pickle

import progressbar as progressbar

from src.configs import path_config
from src.constants import data_type, feature_type
from src.features.lfcc_extractor import extract_lfcc_features


def extract_features(
    part,
    feature,
    access_type="LA",
    proto_files_path=path_config.PROTOCOL_PATH,
    voice_files_path="/Volumes/T7/dataset/LA"
):
    print(f"{part}")
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
    pbar = progressbar.ProgressBar(maxval=len(audio_info))
    pbar.start()
    for i, info in enumerate(audio_info):
        speaker, filename, _, tag, label = info
        audio_path = os.path.join(
            voice_files_path,
            "ASVSpoof2019_" + access_type + "_" + part,
            "flac",
            filename + ".flac"
        )
        if feature == feature_type.LFCC:
            features = extract_lfcc_features(audio_path)
        else:
            features = None
        with open(
            os.path.join(
                path_config.FEATURE_PATH,
                part,
                feature,
                filename + '.pkl'
            ),
            "wb"
        ) as f:
            pickle.dump(features.numpy(), f)
        pbar.update(i + 1)
    pbar.finish()


if __name__ == "__main__":
    extract_features(data_type.TRAIN, feature_type.LFCC)
    extract_features(data_type.DEV, feature_type.LFCC)
    extract_features(data_type.EVAL, feature_type.LFCC)
