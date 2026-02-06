"""
PyTorch integration examples.

This example demonstrates zero-copy tensor conversion and integration
with PyTorch DataLoader for training neural networks on biological sequences.

Requirements:
    pip install bioenc[torch]
    # or
    pip install torch
"""

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    print("or: pip install bioenc[torch]")
    exit(1)

import bioenc

# Example 1: Basic NumPy to PyTorch conversion
print("=" * 60)
print("Example 1: Zero-copy NumPy to PyTorch conversion")
print("=" * 60)

sequence = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
tokens = bioenc.tokenize_dna(sequence, k=3)

# Convert to PyTorch tensor (zero-copy when device='cpu')
tensor = bioenc.to_torch(tokens, device='cpu')

print(f"NumPy tokens: {tokens}")
print(f"PyTorch tensor: {tensor}")
print(f"Tensor dtype: {tensor.dtype}")
print(f"Tensor device: {tensor.device}")
print(f"Zero-copy: {tensor.data_ptr() == tokens.ctypes.data}")
print()

# Example 2: Batch conversion with padding
print("=" * 60)
print("Example 2: Batch conversion with padding")
print("=" * 60)

sequences = [
    b"ATGCGTAAATGA",    # 4 codons
    b"TTTGGG",          # 2 codons
    b"AAACCCGGGTTT",    # 4 codons
]

# Tokenize batch
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

tokens_list = bioenc.batch_tokenize_dna_shared(buffer, lengths, k=3)

# Convert to padded PyTorch tensor
padded_tensor, seq_lengths = bioenc.batch_to_torch(
    tokens_list,
    pad_value=-1,
    device='cpu'
)

print(f"Batch size: {padded_tensor.shape[0]}")
print(f"Max sequence length: {padded_tensor.shape[1]}")
print(f"Padded tensor:\n{padded_tensor}")
print(f"Sequence lengths: {seq_lengths}")
print(f"\nUse lengths for attention masks or PackedSequence")
print()

# Example 3: Custom Dataset for DNA sequences
print("=" * 60)
print("Example 3: Custom PyTorch Dataset")
print("=" * 60)


class DNADataset(Dataset):
    """PyTorch Dataset for DNA sequences with bioenc tokenization."""

    def __init__(self, sequences, k=3, stride=3):
        """
        Args:
            sequences: List of DNA sequences (bytes)
            k: K-mer size (default: 3 for codons)
            stride: Stride (default: 3 for non-overlapping)
        """
        self.sequences = sequences
        self.k = k
        self.stride = stride

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = np.frombuffer(self.sequences[idx], dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=self.k, stride=self.stride)
        # Convert to torch tensor
        return torch.from_numpy(tokens)


# Create dataset
sequences = [
    b"ATGCGTAAATGA",
    b"TTTGGGCCC",
    b"AAACCCGGGTTT",
    b"GCGTACGTA",
]

dataset = DNADataset(sequences, k=3)

print(f"Dataset size: {len(dataset)}")
print(f"First sample: {dataset[0]}")
print(f"Sample type: {type(dataset[0])}")
print()

# Example 4: DataLoader with collate function for padding
print("=" * 60)
print("Example 4: DataLoader with padding collate function")
print("=" * 60)


def collate_dna(batch):
    """Collate function to pad variable-length sequences."""
    # Find max length in batch
    max_len = max(len(item) for item in batch)

    # Pad sequences
    padded = torch.full((len(batch), max_len), -1, dtype=torch.long)
    lengths = torch.tensor([len(item) for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        padded[i, :len(item)] = item

    return padded, lengths


# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_dna
)

print("DataLoader batches:")
for batch_idx, (padded_batch, lengths) in enumerate(dataloader):
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Shape: {padded_batch.shape}")
    print(f"  Lengths: {lengths}")
    print(f"  Padded batch:\n{padded_batch}")
print()

# Example 5: All reading frames for multi-frame models
print("=" * 60)
print("Example 5: All reading frames to PyTorch (6-frame analysis)")
print("=" * 60)

sequences = [
    b"ATGCGTAAATGA",
    b"TTTGGGCCCAAA",
]

buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

# Get all 6 frames for each sequence
all_frames = bioenc.batch_tokenize_dna_all_frames(
    buffer, lengths, k=3
)

# Convert to PyTorch tensor [batch_size, 6, max_len]
frames_tensor = bioenc.frames_to_torch(all_frames, pad_value=-1, device='cpu')

print(f"Frames tensor shape: {frames_tensor.shape}")
print(f"  Dimension 0: batch size ({frames_tensor.shape[0]})")
print(f"  Dimension 1: number of frames ({frames_tensor.shape[1]})")
print(f"  Dimension 2: max sequence length ({frames_tensor.shape[2]})")
print(f"\nSequence 0, Frame 0: {frames_tensor[0, 0]}")
print(f"Sequence 0, Frame 3 (reverse): {frames_tensor[0, 3]}")
print()

# Example 6: Simple embedding model
print("=" * 60)
print("Example 6: Simple DNA embedding model")
print("=" * 60)


class SimpleDNAModel(nn.Module):
    """Simple embedding model for DNA sequences."""

    def __init__(self, vocab_size=125, embedding_dim=64, hidden_dim=128):
        """
        Args:
            vocab_size: Size of k-mer vocabulary (5^3 = 125 for k=3, ACGTN)
            embedding_dim: Dimension of k-mer embeddings
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=-1  # Padding value
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, x, lengths):
        """
        Args:
            x: Padded token sequences [batch_size, seq_len]
            lengths: Original sequence lengths [batch_size]

        Returns:
            Logits [batch_size, 2]
        """
        # Clamp negative padding values to 0 for embedding
        x_clamped = x.clamp(min=0)

        # Embed tokens
        embedded = self.embedding(x_clamped)  # [batch_size, seq_len, embedding_dim]

        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)

        # Use final hidden state
        logits = self.fc(hidden[-1])  # [batch_size, 2]

        return logits


# Create model
model = SimpleDNAModel(vocab_size=125, embedding_dim=64, hidden_dim=128)

# Example forward pass
sequences = [b"ATGCGTAAATGA", b"TTTGGGCCC"]
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths_np = np.array([len(s) for s in sequences], dtype=np.int64)

tokens_list = bioenc.batch_tokenize_dna_shared(buffer, lengths_np, k=3)
padded, lengths_torch = bioenc.batch_to_torch(tokens_list, pad_value=-1, device='cpu')

print(f"Model: {model.__class__.__name__}")
print(f"Input shape: {padded.shape}")
print(f"Lengths: {lengths_torch}")

with torch.no_grad():
    output = model(padded, lengths_torch)

print(f"Output shape: {output.shape}")
print(f"Output logits:\n{output}")
print()

# Example 7: GPU acceleration
print("=" * 60)
print("Example 7: GPU acceleration (if available)")
print("=" * 60)

if torch.cuda.is_available():
    print("CUDA is available!")
    device = 'cuda'

    # Tokenize on CPU
    sequence = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
    tokens = bioenc.tokenize_dna(sequence, k=3)

    # Convert to CUDA tensor
    tensor_gpu = bioenc.to_torch(tokens, device='cuda')

    print(f"Tensor on GPU: {tensor_gpu.device}")
    print(f"Tensor: {tensor_gpu}")

    # Move model to GPU
    model_gpu = model.to('cuda')
    padded_gpu = bioenc.to_torch(
        tokens_list[0], device='cuda'
    ).unsqueeze(0)  # Add batch dimension
    lengths_gpu = torch.tensor([len(tokens_list[0])], device='cuda')

    with torch.no_grad():
        output_gpu = model_gpu(padded_gpu, lengths_gpu)

    print(f"Output from GPU: {output_gpu}")
else:
    print("CUDA not available. Running on CPU.")
    print("To use GPU: Install CUDA-enabled PyTorch")
print()

print("=" * 60)
print("Summary of PyTorch Integration Features")
print("=" * 60)
print("1. to_torch() - Zero-copy NumPy to PyTorch conversion")
print("2. batch_to_torch() - Batch conversion with automatic padding")
print("3. frames_to_torch() - Multi-frame tensor conversion")
print("4. Compatible with PyTorch Dataset and DataLoader")
print("5. Support for CPU and CUDA devices")
print("6. Efficient padding for variable-length sequences")
print("7. Ready for embedding layers and neural networks")
