class ConvNeuralNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnnLayers = nn.Sequential(

            # First convolutional layer
            nn.Conv2d(1, 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Second convolutional layer
            nn.Conv2d(4, 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.linearLayers = nn.Sequential(nn.Linear(4 * 7 * 7, 36))
    
    # feed-forward
    def forward(self, input):

        input = self.cnnLayers(input)
        input = input.view(input.size(0), -1)
        input = self.linearLayers(input)

        return input

        