import torch
import torch.nn.functional as f
import torch.nn as nn


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
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
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
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.avg_pool = nn.MaxPool2d(2)

        self.linear_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1000),
        )

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        y = self.conv_layer_1.forward(x)

        y = self.conv_layer_2.forward(y)

        y = self.conv_layer_3.forward(y)

        y = self.conv_layer_4.forward(y)

        y = self.conv_layer_5.forward(y)

        y = self.avg_pool.forward(y)

        y = y.reshape(len(x), -1)

        y = self.linear_layer(y)

        return y

    def classify(self, x):
        scores = self.forward(x)
        classes = torch.argmax(scores, dim=1)

        return classes
    
    def epoch_train(self, i_epoch, dataloader_train, dataloader_test, optim, criterion):

        self.train()
        for i_batch, (x, y) in enumerate(dataloader_train):
            
            optim.zero_grad()

            y_predicted = self.forward(x)
            
            loss = criterion.forward(y_predicted, y)

            if i_batch % 100 == 0:
                self.eval()
                with torch.no_grad():
                    test_acc = self.get_classification_error(dataloader_test) * 100
                    print(f"epoch {i_epoch}, batch {i_batch} - loss {loss.item()}, test acc: {test_acc:.2f}")
                self.train()

            loss.backward()

            optim.step()
            
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
