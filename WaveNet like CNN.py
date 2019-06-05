class CNN_Model(nn.Module):
    def __init__(self, input_size, output_size):

        super(self.__class__, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(output_size, output_size, 11, stride=1, dilation=2, groups=1, bias=True)
        self.conv2 = torch.nn.Conv1d(output_size, output_size, 11, stride=1, dilation=1, groups=1, bias=True) #20
        self.conv3 = torch.nn.Conv1d(output_size, output_size, 9, stride=1, dilation=1, groups=1, bias=True) #12
        self.conv4 = torch.nn.Conv1d(output_size, output_size, 7, stride=1, dilation=1, groups=1, bias=True) #6
        self.conv5 = torch.nn.Conv1d(output_size, output_size, 5, stride=1, dilation=1, groups=1, bias=True) #2
        self.fc = torch.nn.Linear(output_size * 2, output_size)
        
        self.softmax = nn.Softmax(dim = 1)
        

    def forward(self, input_sequences, hidden=None): #hidden remains for easier code compatibility
        
        batch_size = input_sequences.shape[0]
        
        out = torch.transpose(input_sequences, 1, 2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.fc(out.view(batch_size, -1))
        out = self.softmax(out)
        
        return out, hidden
