"""
This script defines Temporal Conv.
"""
import torch.nn as nn

class TCR(nn.Module):
    def __init__(self, depth=3, kernel_size=3, stride=1, padding = 1):
        super(TCR, self).__init__()
        in_channels, out_channels = 512, 512
        Temporal_Block1 = nn.Sequential(
                nn.Conv1d(16, 16, kernel_size, stride=stride, padding = padding),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(),
                )
        Temporal_Block2 = nn.Sequential(
                nn.Conv1d(16, out_channels, kernel_size, stride=stride, padding = padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(),
                )
        Temporal_Block3 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding = padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(),
                )        
        layers = []
        layers.append(Temporal_Block1)
        layers.append(Temporal_Block2)
        for i in range(depth):
            layers.append(Temporal_Block3)
        self.TCR = nn.Sequential(*layers)


    def forward(self, x):
        # (V*T, c, h, w)
        return self.TCR(x)    output=se(input)
    print(output.shape)