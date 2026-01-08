import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder, vocabulary):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocabulary = vocabulary
    
    def forward(self, X, caption, teacher_forcing_ratio=0.5):
        X = self.encoder(X)
        X = self.decoder(X, caption, teacher_forcing_ratio)
        return X
    
    def generate(
        self,
        images,
        max_len=20,
        beam_size=5
    ):
        encoder_out = self.encoder(images)

        generated_captions = batch_beam_search(
            self.decoder,
            encoder_out,
            sos_idx=self.vocabulary.stoi["<SOS>"],
            eos_idx=self.vocabulary.stoi["<EOS>"],
            max_len=max_len,
            beam_size=beam_size
        )

        sentences = [
            " ".join(
                self.vocabulary.itos[idx]
                for idx in caption
            )
            for caption in generated_captions
        ]
        
        return sentences


def beam_search(
    decoder,
    encoder_out,
    sos_idx,
    eos_idx,
    max_len=20,
    beam_size=5,
    length_penalty=0.7
):
    """
    encoder_out: (1, encoder_dim)
    """

    device = encoder_out.device

    # Initialize hidden state
    h = decoder.init_h(encoder_out).unsqueeze(0)  # (1, 1, hidden_dim)
    c = decoder.init_c(encoder_out).unsqueeze(0)

    # Beams: (sequence, score, hidden, cell)
    beams = [(
        [sos_idx],
        0.0,
        h,
        c
    )]

    completed = []

    for _ in range(max_len):
        new_beams = []

        for seq, score, h, c in beams:
            if seq[-1] == eos_idx:
                completed.append((seq, score))
                continue

            input_token = torch.tensor([[seq[-1]]], device=device)
            embed = decoder.embed(input_token)

            output, (h_new, c_new) = decoder.lstm(embed, (h, c))
            logits = decoder.fc(output.squeeze(1))
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for i in range(beam_size):
                new_seq = seq + [topk_indices[0, i].item()]
                new_score = score + topk_log_probs[0, i].item()

                new_beams.append((
                    new_seq,
                    new_score,
                    h_new,
                    c_new
                ))

        beams = sorted(
            new_beams,
            key=lambda x: x[1] / (len(x[0]) ** length_penalty),
            reverse=True
        )[:beam_size]

        if not beams:
            break

    completed.extend(beams)

    best_seq = max(
        completed,
        key=lambda x: x[1] / (len(x[0]) ** length_penalty)
    )[0]

    return best_seq
    
def batch_beam_search(
    decoder,
    encoder_outs,
    sos_idx,
    eos_idx,
    max_len=20,
    beam_size=5
):
    results = []

    for i in range(encoder_outs.size(0)):
        seq = beam_search(
            decoder,
            encoder_outs[i:i+1],
            sos_idx,
            eos_idx,
            max_len=max_len,
            beam_size=beam_size
        )
        results.append(seq)

    return results

