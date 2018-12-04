import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torchvision.models as models
import math

class XRayModel(nn.Module):
    def __init__(self, num_outputs=2):
        super(XRayModel,self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(32), stride=(1), padding=(1), bias=False)
        self.dense_layers = models.densenet121(pretrained=True)

        self.all_cnn = nn.Sequential(
                nn.Dropout(0.2),
                nn.Conv2d(1,96,(3,3), padding=1), # changed to 1 input channel
                nn.ReLU(),
                nn.Conv2d(96,96,(3,3), padding=1),
                nn.ReLU(),
                nn.Conv2d(96,96,(3,3), padding=1, stride=2),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Conv2d(96,192,(3,3), padding=1),
                nn.ReLU(),
                nn.Conv2d(192,192,(3,3), padding=1),
                nn.ReLU(),
                nn.Conv2d(192,192,(3,3), padding=1, stride=2),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Conv2d(192,192,(3,3)),
                nn.ReLU(),
                nn.Conv2d(192,192,(1,1)),
                nn.ReLU(),
                nn.Conv2d(192,10,(1,1)),
                nn.ReLU(),
                nn.AvgPool2d((6,6)))
                # normalize??
        self.classifier1 = nn.Linear(in_features=1000, out_features=256)
        self.classifier2 = nn.Linear(in_features=256, out_features=num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            #elif isinstance(m, nn.Conv2d):
            #    nn.init.kaiming_normal_(m.weight, mode='fan_out')

        # initialize model


    def forward(self, batch_input):
        #out = self.dense_layers(batch_input)
        #out = self.all_cnn(batch_input)
        out = self.conv1(batch_input)
        out = self.dense_layers(out)
        #import pdb
        #pdb.set_trace()
        out = out.view(out.shape[0], -1)
        #out = self.dense_layers(batch_input)
        out = self.classifier1(out)
        out = self.classifier2(out)
        return out

