import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torchvision.models as models
import math

class XRayModel(nn.Module):
    def __init__(self, num_outputs=2):
        super(XRayModel,self).__init__()
        #self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.dense_layers = models.densenet121(pretrained=True)
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
        self.classifier = nn.Linear(in_features=17640, out_features=num_outputs)

        # initialize model


    def forward(self, batch_input):
        #out = self.dense_layers(batch_input)
        out = self.all_cnn(batch_input)
        #import pdb
        #pdb.set_trace()
        out = out.view(out.shape[0], -1)
        #out = self.dense_layers(batch_input)
        out = self.classifier(out)
        return out

