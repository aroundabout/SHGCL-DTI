from torch.utils.data import Dataset


class DTIDataSet(Dataset):
    def __init__(self, feature, label):
        self.data = feature
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)
