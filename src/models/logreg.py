import torch



class LogReg(torch.nn.Module):
    def __init__(self, in_features, K, num_classes):
        super(LogReg, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.client_bias = torch.nn.Embedding(K, in_features)

        self.classifier = torch.nn.Sequential(
            # torch.nn.Linear(in_features, in_features, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Linear(in_features, num_classes, bias=True),
            # torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, b):
        # print(b)
        bias = self.client_bias(b)

        # x = self.classifier(x + bias)
        # x = self.classifier(bias)
        x = self.classifier(x)
        return x
