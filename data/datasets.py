from __future__ import annotations
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import pickle
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

class B3ESMDataset(Dataset):
    def __init__(self, split='train', augs=['mean', 'smote', 'pca'], pca_model=None, dtype=torch.float32, data_folder='/data/b3esm/', file_start='data_'):
        self.split = split
        self.augs = augs
        self.smote = SMOTE(random_state=42)
        self.pca_model = pca_model

        self.dataset = pickle.load(open(os.path.join(data_folder, file_start+split+".pkl"), "rb"))

        self.labels = torch.tensor(self.dataset['Number'].values).reshape(-1, 1).to(dtype)

        if not isinstance(augs, list): augs = [augs]

        self.embeds = None

        for aug in augs:
            if isinstance(aug, str):
                assert aug.lower() in ['mean', 'flatten', 'smote', 'pca'], f"if aug={aug} is a string, it must be in [\'mean\', \'flatten\', \'smote\', \'pca\']"
                if aug == 'mean':
                    assert self.embeds is None
                    self.embeds = torch.tensor([e.mean(axis=1).reshape(-1,) for e in self.dataset['Embeddings']]).to(dtype)
                elif aug == 'flatten':
                    assert self.embeds is None
                    max_T = max([e.shape[1] for e in self.dataset['Embeddings']])
                    padded_embeddings = [torch.tensor(np.pad(e, ((0, 0), (0, max_T - e.shape[1]), (0, 0)))).reshape(1, -1) for e in self.dataset['Embeddings']]
                    self.embeds = torch.concatenate(padded_embeddings, dim=0).to(dtype)
                elif aug == 'smote':
                    if split != 'train': continue # only uses smote on train set
                    assert self.embeds is not None # needs embeds to not be None
                    embeds_smote, labels_smote = self.smote.fit_resample(self.embeds.numpy(), self.labels.numpy())
                    self.embeds = torch.tensor(embeds_smote).to(dtype)
                    self.labels = torch.tensor(labels_smote).to(dtype)
                elif aug == 'pca':
                    assert self.pca_model is not None and self.embeds is not None and (split == 'train' or hasattr(self.pca_model, "components_")), "pca_model should be given and if split isn't trained, it needs to be pre-fitted"
                    if split == 'train' and not hasattr(self.pca_model, "components_"):
                        self.pca_model.fit(self.embeds.numpy())
                    self.embeds = torch.tensor(self.pca_model.transform(self.embeds.numpy())).to(dtype)
            else:
                self.embeds, self.labels = aug(self.embeds, self.labels)
        assert self.embeds is not None and self.labels is not None, f"embeds={self.embeds}\nlabels={self.labels}"

    def __getitem__(self, index):
       return (self.embeds[index], self.labels[index])

    def __len__(self):
        return len(self.labels)