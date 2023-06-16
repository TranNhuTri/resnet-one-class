import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
from torch.utils.data.dataloader import default_collate

from configs import model_config
from constants import data_type

torch.set_default_tensor_type(torch.FloatTensor)


def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padding_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padding_len, dtype=spec.dtype)), 1)


def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


class ASVSpoof2019(Dataset):
    def __init__(
        self,
        access_type,
        path_to_features,
        path_to_protocol,
        part="train",
        feature_type="lfcc",
        genuine_only=False,
        feat_len=model_config.FEAT_LEN,
        padding_type='repeat'
    ):
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature_type = feature_type
        self.path_to_protocol = path_to_protocol
        self.padding_type = padding_type
        if self.part == data_type.TRAIN:
            proto_file_extension = ".trn.txt"
        else:
            proto_file_extension = ".trl.txt"
        protocol = os.path.join(
            self.path_to_protocol,
            "ASVSpoof2019." + access_type + ".cm." + self.part + proto_file_extension
        )
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8,
                        "A09": 9, "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16,
                        "A17": 17, "A18": 18, "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}

        self.label = {"spoof": 1, "bonafide": 0}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in [data_type.TRAIN, data_type.DEV]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        try:
            with open(self.ptf + '/' + filename + self.feature_type + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            with open(
                os.path.join(self.path_to_features, self.part) + '/' + self.feature_type + '/' + filename + '.pkl',
                "rb"
            ) as feature_handle:
                feat_mat = pickle.load(feature_handle)

        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            start_p = np.random.randint(this_feat_len - self.feat_len)
            feat_mat = feat_mat[:, start_p:start_p + self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding_type == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding_type == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('padding should be zero or repeat!')

        return feat_mat, filename, self.tag[tag], self.label[label]

    @staticmethod
    def collate_fn(samples):
        return default_collate(samples)
