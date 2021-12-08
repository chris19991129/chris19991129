import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import re

import pandas as pd

import argparse

from networks import *
from dataloaders import dataset
from misc import progress_bar

CLASSES = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')

def main():
    parser = argparse.ArgumentParser(description="Classification with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--testBatchSize', default=16, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--resume', type=str,  default='./checkpoints/model.pth', help='the path of pretrained model to resume')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    solver = Solver(args)
    solver.run()

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.test_batch_size = config.testBatchSize
        self.resume = config.resume
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.test_loader = None

    def load_data(self):
        test_transform = transforms.Compose([transforms.ToTensor()])
        #test_set = dataset.Classification_Dataset(root_dir='./raw_images/val',  csv_file='./raw_images/val_labels.csv',transform=test_transform)
        test_set = dataset.Test_Dataset(root_dir='./raw_images/test', transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        self.model = densenet121(pretrained=False, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)
        # self.model = DenseNet161(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)
        # self.model = DenseNet169(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)
        # self.model = DenseNet201(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)

        state_dict = torch.load(self.resume).state_dict()
        self.model.load_state_dict(state_dict)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = torch.argmax(target, dim=1)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total
    
    def predict(self):
        print("test:")
        self.model.eval()

        result = [];
        resultID = [];

        with torch.no_grad():
            for batch_num, (data, ID) in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                prediction = torch.max(output, 1)
                result.append(prediction[1].cpu().numpy().maxarg())
                resultID.append(ID)
                progress_bar(batch_num, len(self.test_loader),"Thanks me for staying sane XD--By Edward")

        return result, resultID
    
    def write_csv_kaggle_sub(self,fname, ID, Y):
        tmp = [['ID', 'label']]
        for (i,y) in enumerate(Y):
            tmp2 = [ID[i], self.num2cat(y)]
            tmp.append(tmp2)
        with open(fname, 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(tmp)

    def run(self):
        train_labels_info = pd.read_csv('raw_images/train_labels.csv', header=0)
        
        target = np.eye(6)[CLASSES[0].astype(int)]
	target = torch.argmax(target, dim=1)
        print(target.cpu().numpy())

        self.load_data()
        self.load_model()
        print("Done Load")
        test_result = self.predict()
        #print(test_result[0])
        #print(type(test_result),type(test_result[0]),type(test_result[1]))
        self.write_csv_kaggle_sub('naive_baseline.csv', test_result[1], test_result[0])
        #print("===> ACC. PERFORMANCE: %.3f%%" % (test_result[1] * 100))

if __name__ == '__main__':
    main()