import os
import pickle
import statistics

import numpy as np
import progressbar as progressbar
from scipy import stats as st

from src.configs import path_config
from src.constants import data_type, feature_type
from src.features.delta_spetral_cepsrtal_extractor import extract_delta_spectral_cepstral_features
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
    feature_shapes = []
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
        elif feature == feature_type.DSCC:
            features = extract_delta_spectral_cepstral_features(audio_path)
        else:
            features = None
        feature_shapes.append(features.shape)
        with open(
            os.path.join(
                path_config.FEATURE_PATH,
                part,
                feature,
                filename + '.pkl'
            ),
            "wb"
        ) as f:
            pickle.dump(features, f)
        pbar.update(i + 1)
    pbar.finish()
    feature_shapes = np.array(feature_shapes)
    mode = st.mode(feature_shapes)
    print("Shape")
    print("Mode:", mode)
    with open(os.path.join(path_config.FEATURE_PATH, f"feature_{part}_shape.txt"), 'w') as feature_shape_file:
        feature_shape_file.write(f"Mode: {str(mode)}")


if __name__ == "__main__":
    # extract_features(data_type.TRAIN, feature_type.LFCC)
    # extract_features(data_type.DEV, feature_type.LFCC)
    # extract_features(data_type.EVAL, feature_type.LFCC)

    extract_features(data_type.TRAIN, feature_type.DSCC)
    extract_features(data_type.DEV, feature_type.DSCC)
    extract_features(data_type.EVAL, feature_type.DSCC)
