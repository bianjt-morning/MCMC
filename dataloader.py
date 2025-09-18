from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

path = './datasets'
class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels
    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()



class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()



class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()





class scene2688(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'scene2688.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'scene2688.mat')['X2'].astype(np.float32)
        self.view3 = scipy.io.loadmat(path+'scene2688.mat')['X3'].astype(np.float32)
        self.view4 = scipy.io.loadmat(path+'scene2688.mat')['X4'].astype(np.float32)
        labels = scipy.io.loadmat(path+'scene2688.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 2688
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])
        , torch.from_numpy(self.view3[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()







class RSSCN7(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'RSSCN7.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'RSSCN7.mat')['X2'].astype(np.float32)
        self.view3 = scipy.io.loadmat(path+'RSSCN7.mat')['X3'].astype(np.float32)
        self.view4 = scipy.io.loadmat(path+'RSSCN7.mat')['X4'].astype(np.float32)
        self.view5 = scipy.io.loadmat(path+'RSSCN7.mat')['X5'].astype(np.float32)
        labels = scipy.io.loadmat(path+'RSSCN7.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 2800
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])
        , torch.from_numpy(self.view3[idx]), torch.from_numpy(self.view4[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class MirFlickr(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'MirFlickr.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'MirFlickr.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'MirFlickr.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 12154
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()



class STL10_deep(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'STL10_deep.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'STL10_deep.mat')['X2'].astype(np.float32)
        self.view3 = scipy.io.loadmat(path+'STL10_deep.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'STL10_deep.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 13000
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])
        , torch.from_numpy(self.view3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CIFAR10_deep(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'CIFAR10_deep.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'CIFAR10_deep.mat')['X2'].astype(np.float32)
        self.view3 = scipy.io.loadmat(path+'CIFAR10_deep.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'CIFAR10_deep.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])
        , torch.from_numpy(self.view3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class Hdigit(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'Hdigit.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'Hdigit.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'Hdigit.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class LabelMe(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'LabelMe.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'LabelMe.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'LabelMe.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 2688
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()



class CIFAR10(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'CIFAR10-view5.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'CIFAR10-view5.mat')['X2'].astype(np.float32)
        self.view3 = scipy.io.loadmat(path+'CIFAR10-view5.mat')['X3'].astype(np.float32)
        self.view4 = scipy.io.loadmat(path+'CIFAR10-view5.mat')['X4'].astype(np.float32)
        self.view5 = scipy.io.loadmat(path+'CIFAR10-view5.mat')['X5'].astype(np.float32)
        labels = scipy.io.loadmat(path+'/CIFAR10-view5.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 60000
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])
        , torch.from_numpy(self.view3[idx]), torch.from_numpy(self.view4[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()



def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./datasets/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "CCV":
        dataset = CCV('./datasets/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./datasets/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10

    elif dataset == "Caltech-2V":
        dataset = Caltech('./datasets/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('./datasets/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('./datasets/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('./datasets/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "CIFAR10-v5":
        dataset = CIFAR10('./datasets/')
        dims = [768, 576, 512, 640, 944]
        view = 5
        data_size = 60000
        class_num = 10

    elif dataset == "scene2688":
            dataset = scene2688('./datasets/')
            dims = [512, 432, 256, 48]
            view = 4
            data_size = 2688
            class_num = 8


    elif dataset == "RSSCN7":
            dataset = RSSCN7('./datasets/')
            dims = [768, 540, 885, 512, 800]
            view = 5
            data_size = 2800
            class_num = 7
    elif dataset == "MirFlickr":
            dataset = MirFlickr('./datasets/')
            dims = [100, 100]
            view = 2
            data_size = 12154
            class_num = 7

    elif dataset == "STL10_deep":
            dataset = STL10_deep('./datasets/')
            dims = [1024, 512, 2048]
            view = 3
            data_size = 13000
            class_num = 10

    elif dataset == "CIFAR10_deep":
            dataset = CIFAR10_deep('./datasets/')
            dims = [512, 2048, 1024]
            view = 3
            data_size = 50000
            class_num = 10
    elif dataset == "Hdigit":
            dataset = Hdigit('./datasets/')
            dims = [784, 256]
            view = 2
            data_size = 10000
            class_num = 10

    elif dataset == "LabelMe":
            dataset = LabelMe('./datasets/')
            dims = [512, 245]
            view = 2
            data_size = 2688
            class_num = 10
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
