from torch.utils.data import DataLoader
from data.datasets import B3ESMDataset
from sklearn.decomposition import PCA
import pickle
import os

def build_loader(config):
    if config.DATA.DATASET == "b3esm":
        n_components = int(config.DATA.PCA.N_COMPS)
        pca_model = PCA(n_components=n_components) if 'pca' in config.AUGS else None
        dataset_train = B3ESMDataset(split='train', augs=config.AUGS, pca_model=pca_model)
        dataset_val = B3ESMDataset(split='val', augs=config.AUGS, pca_model=pca_model)
        if pca_model is not None:
            pickle.dump(pca_model, open(os.path.join(config.OUTPUT, "pca_model.pkl"), "wb"))
        if dataset_train.smote is not None:
            pickle.dump(dataset_train.smote, open(os.path.join(config.OUTPUT, "smote.pkl"), "wb"))
        
        # get num_inputs
        embeddings, labels = dataset_train[0]
        num_inputs = int(embeddings.shape[-1])
    else:
        raise NotImplementedError

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True, 
        drop_last=False, 
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
    )

    return num_inputs, dataset_train, dataset_val, data_loader_train, data_loader_val