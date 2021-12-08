import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import re


import pandas as pd
import array as arr
import csv
from sklearn import *

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

    def load_data(self, Train):
        test_transform = transforms.Compose([transforms.ToTensor()])
        if Train==0:
            test_set = dataset.Classification_Dataset(root_dir='./raw_images/train',  csv_file='./raw_images/train_labels.csv',transform=test_transform)
        if Train==1:
            test_set = dataset.Classification_Dataset(root_dir='./raw_images/val',  csv_file='./raw_images/val_labels.csv',transform=test_transform)
        if Train==2:
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
        
        result = [];
        resultType = [];


        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                #print(target)
                target = torch.argmax(target, dim=1)
                output = self.model(data)
               
                #print(output.cpu().numpy().tolist())
                result.extend(output.cpu().numpy().tolist())
                #print(len(result),",",(result[len(result)-1]))
                #print(target.cpu().numpy().tolist())
                resultType.extend(target.cpu().numpy().tolist())
                #print(len(resultType),",",(resultType[len(resultType)-1]))
                progress_bar(batch_num, len(self.test_loader),"Thanks me for staying sane XD--By Edward")
        #print((result),(resultID))
        return result, resultType
    
    def predict(self):
        print("test:")
        self.model.eval()

        result = [];
        resultID = [];

        with torch.no_grad():
            for batch_num, (data, ID) in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                #print(output[1].cpu().numpy().tolist())
                result.extend(output.cpu().numpy().tolist())
                #print(ID.numpy().tolist())
                resultID.extend(ID.numpy().tolist())
                progress_bar(batch_num, len(self.test_loader),"Thanks me for staying sane XD--By Edward")
        #print(len(result),len(result[0]),len(resultID),len(resultID[0]))
        #print((result),(resultID))
        return result, resultID
    
    def write_csv_kaggle_sub(self,fname, ID, Y):
        tmp = [['ID', 'label']]
        for i in range(len(Y)):
            tmp2 = [ID[i], self.num2cat[Y[i]]]
            tmp.append(tmp2)
        with open(fname, 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(tmp)

    def run(self):
        train_labels_info = pd.read_csv('raw_images/train_labels.csv', header=0)  
        categories = list(np.unique(train_labels_info['label']))
        self.num2cat = dict(zip(range(len(categories)), categories))
        self.cat2num = dict(zip(categories,range(len(categories))))
        
        # target = np.array([0,1,2,3,4,5])
        # target = np.eye(6)[target.reshape(-1)]
        # print(target)
        # target = torch.FloatTensor(target[0])
        # target = torch.argmax(target, dim=1)
        # print(target.cpu().numpy())

        self.load_data(0)
        self.load_model()
        print("Done Load")
        self.train_result = self.test()
        print(len(self.train_result[0]),",",len(self.train_result[1]))
        self.load_data(1)
        self.test_result = self.test()
        print(len(self.test_result[0]),",",len(self.test_result[1]))
        for n in [512]:
             for k in [32,64,128]:
                     pca = decomposition.KernelPCA(n_components=n,kernel='rbf')
                     trainW = pca.fit_transform(self.train_result[0])
                     svmclf  =svm.SVC(kernel='rbf',C=k,tol=0.0001)
                     svmclf.fit(trainW, self.train_result[1])
                     valW  = pca.transform(self.test_result[0])
                     predY_svm = svmclf.predict(valW)
                     acc_svm = metrics.accuracy_score(self.test_result[1], predY_svm)
                     print("Kernel PCA svm validation accuracy =", acc_svm)
        #print(test_result[0])
        #print(type(test_result),type(test_result[0]),type(test_result[1]))
                
        Fulltrain = self.train_result[0]+self.test_result[0];
        Lbl = self.train_result[1]+self.test_result[1];
        pca = decomposition.KernelPCA(n_components=512,kernel='rbf')
        trainW = pca.fit_transform(Fulltrain)
        svmclf  =svm.SVC(kernel='rbf',C=64,tol=0.0001)
        svmclf.fit(trainW, Lbl)
        
        self.load_data(2)    
        self.fin_result = self.predict()
        valW  = pca.transform(self.fin_result[0])
        pred = svmclf.predict(valW)
        self.write_csv_kaggle_sub('naive_baseline'+str(512)+'.csv', self.fin_result[1], pred)
        
        #print("===> ACC. PERFORMANCE: %.3f%%" % (test_result[1] * 100))
        exit()

if __name__ == '__main__':
    main()