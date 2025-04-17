import torch
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
from utils import ResBlock
from torchvision.models import resnet50, ResNet50_Weights, ResNet, VGG, vgg11, VGG11_Weights



class IClassifier(nn.Module):
    
    def get_model_name(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def epoch_train(self, i_epoch, dataloader_train, dataloader_test, optim, criterion):
        
        self.train()
        
        for i_batch, (x, y) in enumerate(dataloader_train):
            
            optim.zero_grad()

            y_predicted = self(x)
            
            loss = criterion(y_predicted, y)

            if i_batch % 5 == 0:
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
    
    def get_competition_error_light(self, dataloader):
        with torch.no_grad():

            res = []
            res_true = []

            for i_batch, (x, y_true) in enumerate(dataloader):

                if i_batch == 10:
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
            
            res = np.vstack(res)
            res_true = np.vstack(res_true)

            with open(f'./tmp/res_{self.get_model_name()}.txt', 'w') as f:
                np.savetxt(f, res, fmt='%s')
            
            with open(f'./tmp/res_{self.get_model_name()}_true.txt', 'w') as f:
                np.savetxt(f, res_true, fmt='%s')
            

            with open(f'./tmp/res_{self.get_model_name()}.txt', 'r') as f:
                entries_pred = list([line.split(' ') for line in f.read().split('\n')])

            with open(f'./tmp/res_{self.get_model_name()}_true.txt', 'r') as f:
                entries_true = f.read().split('\n')

            num_corr = 0
            num_total = 0

            for preds, true in zip(entries_pred, entries_true):
                if preds == '' or true == '':
                    break

                if true in preds:
                    num_corr += 1
                num_total += 1
            
            return num_corr / num_total  

    def save_training_progress(self, epoch, optim, loss):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss,
            }, 
            './saved_models/' + self.get_model_name()
        )

    def load_training_progress(self, epoch, optim, loss):
        checkpoint = torch.load('./saved_models/' + self.get_model_name(), weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

        return epoch, optim, loss



class CustomClassifier(IClassifier):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

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

    def get_model_name(self):
        return "custom_classifier"

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



class Resnet50BasedClassifier(IClassifier):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.resnet : ResNet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.c = list(self.resnet.children())[:-1]

        self.linear = nn.Sequential(nn.Linear(2048, 1000), nn.Linear(1000, 1000))

    def get_classifier_name(self):
        return "resnet50_based_classifier"

    def forward(self, x: torch.Tensor):

        with torch.no_grad():
            y = x
            for m in self.c:
                y = m(y)

        y = y.reshape(-1, 2048)
        
        y = self.linear(y)

        return y
    


class VGG11BasedClassifier(IClassifier):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.vgg = vgg11(weights=VGG11_Weights)

        self.conv_network = list(self.vgg.children())[0]

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        self.linear_network = nn.Sequential(
            nn.Linear(12800, 4096),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(2048, 1000)
        )

    def get_model_name(self):
        return "vgg11_based_classifier"

    def forward(self, x):
        
        with torch.no_grad():
            y = self.conv_network(x)

        y = self.adaptive_pool(y)
        
        y = y.flatten(1)

        y = self.linear_network(y)

        return y
