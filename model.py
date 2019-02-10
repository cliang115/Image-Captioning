import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions[:, :-1])
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = self.fc(lstm_out)
        return lstm_out
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sample_out = []
        for i in range(max_len):
            lstm_output, states = self.lstm(inputs, states)
            lstm_output = self.fc(lstm_output.squeeze(1))
            indexes = lstm_output.max(1)[1]
            sample_out.append(indexes)
            inputs = self.word_embeddings(indexes).unsqueeze(1)
        out = torch.stack(sample_out, 1)
        return out.data.tolist()[0]