import random
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
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=8,
                            batch_first=True
                        )
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        
    def forward(self, encoder_out, captions, teacher_forcing_ratio=0.5):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        captions: (B, seq_len)
        """
        device = encoder_out.device
        batch_size, seq_len = captions.size()
        vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, seq_len, vocab_size, device=device)

        # Initial LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c = self.init_c(mean_encoder_out).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        hidden = (h, c)

        current_word = captions[:, 0]
        
        for t in range(1, seq_len):
            embeddings = self.dropout(self.embed(current_word))
            
            query = hidden[0][-1].unsqueeze(1)
            
            context, _ = self.attention(
                query=query,
                key=encoder_out,
                value=encoder_out
            )
            
            context = context.squeeze(1)
            
            lstm_input = torch.cat([embeddings, context], dim=1)
            lstm_input = lstm_input.unsqueeze(1)
            
            output, hidden = self.lstm(lstm_input, hidden)
            
            logits = self.fc(output.squeeze(1))
            outputs[:, t] = logits
            
            use_teacher = random.random() < teacher_forcing_ratio
            current_word = captions[:, t] if use_teacher else logits.argmax(dim=1)

        return outputs