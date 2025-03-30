import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
from utils import ResBlock


class ImageNetClassifier(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.max_pool = nn.MaxPool2d(2)

        self.linear_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1000),
        )

        self.skip_connection_1 = nn.Conv2d(3, 64, 3, 8, 1)
        self.skip_connection_2 = nn.Conv2d(64, 256, 3, 4, 1)
        self.skip_connection_3 = nn.Conv2d(256, 512, 3, 4, 1)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        # N, 3, 224, 224

        y1 = self.conv_layer_1(x)
        
        # N, 64, 56, 56

        y2 = self.conv_layer_2(y1)
        
        # N, 64, 28, 28

        y2 = y2 + self.skip_connection_1(x)

        # N, 64, 28, 28

        y3 = self.conv_layer_3(y2)

        # N, 128, 14, 14

        y4 = self.conv_layer_4(y3)
        
        # N, 256, 7, 7

        y4 = y4 + self.skip_connection_2(y2)

        # N, 256, 7, 7

        y5 = self.conv_layer_5(y4)
        
        # N, 512, 4, 4

        y6 = self.max_pool(y5)
        
        # N, 512, 2, 2

        y6 = y6 + self.skip_connection_3(y4)
        
        # N, 512, 2, 2

        y7 = y6.reshape(len(x), -1)

        # N, 2048

        y8 = self.linear_layer(y7)

        # N, 1000

        return y8

    def classify(self, x):
        scores = self(x)
        classes = torch.argmax(scores, dim=1)

        return classes
    
    def epoch_train(self, i_epoch, dataloader_train, dataloader_test, optim, criterion):

        self.train()
        for i_batch, (x, y) in enumerate(dataloader_train):
            
            optim.zero_grad()

            y_predicted = self(x)
            
            loss = criterion(y_predicted, y)

            if i_batch % 100 == 0:
                self.eval()
                with torch.no_grad():
                    if i_batch % 500 == 0:
                        test_acc = self.get_competition_error_light(dataloader_test) * 100
                    else:
                        test_acc = -1
                    print(f"epoch {i_epoch:3}, batch {i_batch:3} - loss {loss.item():.2f}, train 5-acc: {test_acc:.2f}")
                self.train()

            loss.backward()

            optim.step()
        
        torch.save(self.state_dict(), f'./saved_models/model_{i_epoch}.pt')
            
    def get_classification_error(self, dataloader, max_i_batch = 2):
        with torch.no_grad():
            num_correct = num_total = 0

            for i_batch, (x, y_true) in enumerate(dataloader):

                if i_batch == max_i_batch:
                    break

                y_predicted = self.classify(x)

                num_correct += (y_predicted == y_true).sum() 
                num_total += len(y_predicted)

                
            
            return num_correct / num_total
        
    def get_competition_error_light(self, dataloader):
        
        with torch.no_grad():

            res = []
            res_true = []

            for i_batch, (x, y_true) in enumerate(dataloader):

                if i_batch == 20:
                    break

                scores = self(x)

                _, indexes = torch.topk(scores, k=5, dim=1)

                res.append(indexes.cpu().numpy())
                res_true.append(y_true.cpu().numpy())
            
            # vstack
            res = np.vstack(res)
            res_true = np.hstack(res_true)

            # count correct
            n_true = 0
            n_total = len(res)

            for i in range(n_total):
                if res_true[i] in res[i, :]:
                    n_true += 1
            
            return n_true / n_total

    
    def get_competition_error(self, dataloader):
        with torch.no_grad():

            res = []
            res_true = []

            for i_batch, (x, y_true) in enumerate(dataloader):

                scores = self(x)

                _, indexes = torch.topk(scores, k=5, dim=1)

                res.append(indexes.cpu().numpy())
                res_true.append(y_true.reshape(-1, 1).cpu().numpy())

                if i_batch % 100 == 0:
                    print(i_batch)
            
            res = np.vstack(res)
            res_true = np.vstack(res_true)

            with open('./res.txt', 'w') as f:
                np.savetxt(f, res, fmt='%s')
            
            with open('./res_true.txt', 'w') as f:
                np.savetxt(f, res_true, fmt='%s')
            

            with open('./res.txt', 'r') as f:
                entries_pred = list([line.split(' ') for line in f.read().split('\n')])

            with open('./res_true.txt', 'r') as f:
                entries_true = f.read().split('\n')

            num_corr = 0
            num_total = 0

            for preds, true in zip(entries_pred, entries_true):
                if preds == '' or true == '':
                    break

                if true in preds:
                    num_corr += 1
                num_total += 1

                if int(true) % 100 == 0:
                    print(true)
            
            print(num_corr / num_total)   
