import torch
import torch.nn as nn

class SignalPurifier(nn.Module):
    def __init__(self, input_features = 3, bottleneck = 1):
        super(SignalPurifier, self).__init__()

        # 1. The Compressor (Squeeze the noise out)
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 8),
            nn.ReLU(),
            nn.Linear(8, bottleneck) # The absolute choke point
        )

        # 2. The Reconstructor (Attempt to rebuild reality)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8),
            nn.ReLU(),
            nn.Linear(8, input_features)
        )

    def forward(self, x):
        # We extract the pure signal, then rebuild it
        latent_alpha = self.encoder(x)
        reconstructed_data = self.decoder(latent_alpha)

        # We return both so we can calculate the reconstruction loss,
        # but the LSTM will only ever see 'latent_alpha'
        return reconstructed_data, latent_alpha