import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, fused_output, target_sequence):
        embedded_sequence = self.embedding(target_sequence)
        lstm_input = torch.cat([fused_output.unsqueeze(1), embedded_sequence[:, :-1, :]], dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        logits = self.fc(lstm_output)
        
        return logits
