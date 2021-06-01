import torch
import torch.nn as nn
import torch.nn.functional as F
import time
model_urls = 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'
class  vgg11(nn.Module):
    def __init__(self, num_classes=4, pic_nums=4,init_weights=True):
        super(vgg11, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        B = x.size(0)
        x = x.view(B * 4, 3, 64, 64)
        x = self.layer1(x)
        low = self.layer2(x)
        mid = self.layer3(low)
        high = self.layer4(mid)
        low=self.avgpool(low)
        mid=self.avgpool(mid)
        high = self.avgpool(high)
        low, mid, high = low.squeeze(), mid.squeeze(), high.squeeze()
        b, _ = low.size()
        low, mid, high = low.view(int(b / 4), -1), mid.view(int(b / 4), -1), high.view(int(b / 4), -1)
        pk = torch.cat((low, mid, high), dim=1)
        return pk

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg11_mv(num_class=10):
    model = vgg11(num_classes=num_class, pic_nums=4)
    return model

if __name__=="__main__":
    model = vgg11(num_classes=10, pic_nums=4)
    in_data = torch.randint(0, 255, (32, 4, 3, 64, 64), dtype=torch.float32)
    print(in_data.size())
    out=model(in_data)
    print(out.size())

