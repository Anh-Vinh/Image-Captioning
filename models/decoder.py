import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        
    def forward(self, features, captions, teacher_forcing_ratio):
        
        batch_size, seq_len = captions.shape
        vocab_size = self.fc.out_features
        outputs = torch.zeros(batch_size, seq_len, vocab_size, device=features.device)
        
        current_word = captions[:, 0]
        h0 = self.init_h(features).unsqueeze(0) # (1, B, hidden_dim)
        c0 = self.init_c(features).unsqueeze(0) # (1, B, hidden_dim)
        hidden = (h0, c0)
        
        for t in range(1, seq_len):
            embeddings = self.embed(current_word)
            embeddings = self.dropout(embeddings)
            embeddings = embeddings.unsqueeze(1)
            
            output, hidden = self.lstm(embeddings, hidden)
            logits = self.fc(output.squeeze(1))
            
            outputs[:, t] = logits
            
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            current_word = captions[:, t] if use_teacher else logits.argmax(dim=1)
            
        return outputs
            
        
    # def forward(self, features, captions):
    #     embeddings = self.embed(captions) # (B, T, embed_dim)
    #     embeddings = self.dropout(embeddings)
        
    #     h0 = self.init_h(features).unsqueeze(0) # (1, B, hidden_dim)
    #     c0 = self.init_c(features).unsqueeze(0) # (1, B, hidden_dim)
        
    #     outputs, _ = self.lstm(embeddings, (h0, c0))
    #     outputs = self.fc(outputs) # (B, T, vocab_size)
        
    #     return outputs
    