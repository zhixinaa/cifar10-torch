

## 网络结构

PyTorch实现

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.globalavgpool = nn.AvgPool2d(32, 32)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x= self.globalavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## 调参

| epoch | learning rate | Accuracy of the network on the 10000 test images |
| ----- | ------------- | ------------------------------------------------ |
| 10    | 0.1           | 59%                                              |
| 10    | 0.01          | 69%                                              |
| 10    | 0.005         | 71%                                              |
| 10    | 0.001         | 69%                                              |
| 20    | 0.1           | 62%                                              |
| 20    | 0.01          | 72%                                              |
| 20    | 0.005         | 73%                                              |
| 20    | 0.001         | 73%                                              |

