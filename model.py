# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# NOTE: TransformerEncoder is from PyTorch, not transformers library
# Using PyTorch's built-in Transformer components

# --- Reusable Building Blocks for the U-Net ---

class ConvBlock(nn.Module):
    """A standard convolutional block: (Conv2d -> BatchNorm2d -> LeakyReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    """A standard transposed convolutional block for the U-Net Decoder."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels is the channels from the deeper layer
        # After upsampling, we get out_channels
        # After concatenation with skip, we get out_channels + skip_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # The conv block takes concatenated channels (out_channels + skip_channels)
        # We'll determine this dynamically in forward pass
        self.out_channels = out_channels
        
    def forward(self, x, skip_x):
        x = self.up(x)  # Upsample: in_channels -> out_channels
        
        # Handle size mismatch
        diff_y = skip_x.size()[2] - x.size()[2]
        diff_x = skip_x.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate with skip connection
        x = torch.cat([skip_x, x], dim=1)  # skip_channels + out_channels
        
        # Create conv block dynamically based on actual concatenated channels
        concat_channels = x.size(1)
        conv_block = ConvBlock(concat_channels, self.out_channels).to(x.device)
        
        return conv_block(x)

# --- Dynamic Chunking Layer (No changes needed here) ---
class DynamicChunkingLayer(nn.Module):
    # This class remains exactly as it was in the full U-Net version.
    # Its complexity is not the primary memory bottleneck.
    def __init__(self, d_model, temperature=1.0, eps=1e-8):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        q = self.q_proj(x)
        k = self.k_proj(x)
        sim = F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)
        p_boundary = 1.0 - sim
        first_boundary = torch.ones(batch_size, 1, device=device)
        p_boundary = torch.cat([first_boundary, p_boundary], dim=1)
        logits = torch.stack([p_boundary, 1.0 - p_boundary], dim=-1)
        boundaries_one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        is_boundary = boundaries_one_hot[..., 0]
        chunk_ids = torch.cumsum(is_boundary, dim=1)
        num_chunks = chunk_ids[:, -1].long()
        chunk_ids_0_indexed = (chunk_ids - 1).long().clamp(min=0)

        # Ensure max_chunks is at least 1 to prevent clamp(max=-1) error.
        max_chunks = max(1, int(num_chunks.max().item()))
        chunk_ids_0_indexed = chunk_ids_0_indexed.clamp(max=max_chunks - 1)
        
        pooled_sums = torch.zeros(batch_size, max_chunks, self.d_model, device=device, dtype=x.dtype)
        chunk_counts = torch.zeros(batch_size, max_chunks, 1, device=device, dtype=x.dtype)

        pooled_sums.scatter_add_(1, chunk_ids_0_indexed.unsqueeze(-1).expand_as(x), x)
        chunk_counts.scatter_add_(1, chunk_ids_0_indexed.unsqueeze(-1), torch.ones_like(x[..., :1]))
        
        # Add epsilon to prevent division by zero, which causes NaNs
        pooled_output = pooled_sums / (chunk_counts + self.eps)
        return pooled_output, num_chunks

# --- The Lightweight Model Architecture ---

class WeakTokIDM(nn.Module):
    """The full, lightweight U-Net based model with Dynamic Chunking and Transformer."""
    def __init__(self, in_channels=1, out_channels=1, d_model=256, n_head=4, num_main_layers=4, dropout_rate=0.1, use_dynamic_chunking=True):
        super().__init__()
        self.use_dynamic_chunking = use_dynamic_chunking

        # --- MODIFICATION 1: Reduced Channel Width ---
        # We start with fewer channels (e.g., 32 instead of 64) to reduce memory.
        # This is the "width" of the model.
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- MODIFICATION 2: Reduced Depth ---
        # We have removed the 4th encoder block (enc4) and the deepest bottleneck_conv.
        # This makes the U-Net shallower and significantly reduces the feature map sizes.
        # The bottleneck now happens after the 3rd layer.
        self.bottleneck_channels = 128

        # --- Bottleneck ---
        self.to_transformer = nn.Linear(self.bottleneck_channels, d_model)
        self.dynamic_chunker = DynamicChunkingLayer(d_model=d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model * 4, 
            dropout=dropout_rate, # Apply the dropout rate here
            batch_first=True
        )
        self.main_network = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_main_layers
        )
        
        self.from_transformer = nn.Linear(d_model, self.bottleneck_channels)
        
        # --- Decoder Path (must be symmetric to the new, shallower Encoder) ---
        # After transformer: 128 channels
        # dec1: 128 -> 64, then concat with skip3 (128) -> 192 -> 64
        self.dec1 = DeconvBlock(128, 64)  # Input: 128, Output: 64
        # dec2: 64 -> 32, then concat with skip2 (64) -> 96 -> 32  
        self.dec2 = DeconvBlock(64, 32)   # Input: 64, Output: 32
        # dec3: 32 -> 32, then concat with skip1 (32) -> 64 -> 32
        self.dec3 = DeconvBlock(32, 32)   # Input: 32, Output: 32
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Reshape input from (B, H, W) to (B, C, H, W) for Conv2d layers
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # --- Encoder Path ---
        skip1 = self.enc1(x)
        x = self.pool1(skip1)
        
        skip2 = self.enc2(x)
        x = self.pool2(skip2)

        skip3 = self.enc3(x)
        x = self.pool3(skip3)
        # No 4th layer
        
        # --- Bottleneck Processing ---
        batch_size, channels, mels, time = x.shape
        x_seq = x.permute(0, 3, 2, 1).reshape(batch_size, time * mels, channels)
        
        x_seq = self.to_transformer(x_seq)

        if self.use_dynamic_chunking:
            chunks, num_chunks = self.dynamic_chunker(x_seq)
            
            max_len = chunks.size(1)
            mask = torch.arange(max_len, device=chunks.device)[None, :] >= num_chunks[:, None]
            
            processed_chunks = self.main_network(chunks, src_key_padding_mask=mask)

            # A more robust upsampling: repeat each chunk embedding by an average chunk length
            # Ensure chunks.size(1) is not zero to avoid division error
            if chunks.size(1) > 0:
                avg_chunk_len = (x_seq.size(1) // chunks.size(1))
                processed_seq = torch.repeat_interleave(processed_chunks, repeats=avg_chunk_len, dim=1)
                
                # Handle remainder to match original sequence length
                remainder = x_seq.size(1) - processed_seq.size(1)
                if remainder > 0:
                    processed_seq = torch.cat([processed_seq, processed_seq[:, -1:].repeat(1, remainder, 1)], dim=1)
                elif remainder < 0:
                    processed_seq = processed_seq[:, :x_seq.size(1), :]
            else: # If no chunks are made, pass an empty sequence
                processed_seq = torch.zeros_like(x_seq)

        else: # Bypass dynamic chunking
            # No chunking, so the sequence length remains the same.
            # No mask is needed for the transformer in this case.
            processed_seq = self.main_network(x_seq)

        processed_seq = self.from_transformer(processed_seq)
        x = processed_seq.reshape(batch_size, time, mels, channels).permute(0, 3, 2, 1)

        # --- Decoder Path with Skip Connections ---
        # Note: The decoder path is now shorter to match the encoder
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        # The DeconvBlock expects two inputs, but the last one has no deeper layer to come from.
        # Let's adjust the final upsampling stage. The last decoder block will take the output of dec2
        # and skip1.
        # This is slightly tricky, let's simplify the last stage.
        # After dec2, x is at the same resolution as skip1.
        x = self.dec3(x, skip1)
        
        output = self.out_conv(x)
        return output

# --- Test Block to verify shapes ---
if __name__ == '__main__':
    # B: batch size, C: channels, M: Mel bins, T: time frames
    test_spectrogram = torch.randn(2, 1, 128, 1024)
    model = WeakTokIDM(in_channels=1, out_channels=1, d_model=192, n_head=4, num_main_layers=3)
    
    print(f"Testing LIGHTWEIGHT model architecture.")
    print("Input shape:", test_spectrogram.shape)
    
    # Check model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {num_params / 1e6:.2f}M")
    
    output_spectrogram = model(test_spectrogram)
    
    print("Output shape:", output_spectrogram.shape)
    assert test_spectrogram.shape == output_spectrogram.shape
    print("\nModel forward pass successful with matching input/output shapes!")