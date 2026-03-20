import torch
import torch.nn as nn
from autoencoder import SignalPurifier

class HybridLSTM(nn.Module):
    # We add bottleneck = 1 to squeeze the 3 features down to a single pure signal 
    def __init__(self, input_size = 3, hidden_size = 64, num_layers=2, bottleneck=1):
        super(HybridLSTM, self).__init__()

        # 1. The Noise Filter
        self.purifier = SignalPurifier(input_features=input_size, bottleneck=bottleneck)

        # 2. The Sequence Engine (Now reading ONLY the pure bottleneck signal)
        self.lstm = nn.LSTM(
            input_size=bottleneck,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # 3. The Final Squeeze
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Features)
        batch_size, seq_len, features = x.size()

        # We must flatten the sequence to shove it through the linear Autoencoder
        flat_x = x.view(-1, features)

        # Purify the signal (we ignore the reconstructed data for now, we just want the pure alpha)
        _, latent_alpha = self.purifier(flat_x)

        # Reshape the pure, filtered signal back into a chronological time sequence
        # Shape becomes: (Batch, Sequence_Length, Bottleneck)
        clean_sequence = latent_alpha.view(batch_size, seq_len, -1)

        # Feed the noise-free sequence into the LSTM
        lstm_out, _ = self.lstm(clean_sequence)

        # LSTMs natively understand time, so we safely grab the final day's context
        final_day = lstm_out[:, -1, :]

        prediction = self.linear(final_day)
        return prediction