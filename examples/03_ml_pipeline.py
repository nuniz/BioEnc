"""
Integration with ML frameworks (PyTorch and TensorFlow).

This example shows how to integrate bioenc with popular deep learning frameworks
for genomics ML pipelines.
"""

import numpy as np
import bioenc

# Generate sample data
print("=" * 60)
print("Preparing sample genomic data")
print("=" * 60)

np.random.seed(42)
num_samples = 100
seq_len = 500
k = 6

# Generate random DNA sequences
bases = np.frombuffer(b"ACGT", dtype=np.uint8)
sequences = [bases[np.random.randint(0, 4, seq_len)] for _ in range(num_samples)]

# Create shared buffer
buffer = np.concatenate(sequences)
lengths = np.full(num_samples, seq_len, dtype=np.int64)

# Tokenize with canonical k-mers
max_len = 100
tokens = bioenc.batch_tokenize_dna_shared(
    buffer, lengths,
    k=k, max_len=max_len, pad_value=-1,
    stride=1, alphabet="acgtn", strand="canonical",
    enable_padding=True  # Return rectangular array
)

print(f"Generated {num_samples} sequences of length {seq_len}")
print(f"Tokenized to shape: {tokens.shape}")
print(f"Vocabulary size: {5**k:,}")
print()

# =============================================================================
# PyTorch Example
# =============================================================================

try:
    import torch
    import torch.nn as nn

    print("=" * 60)
    print("PyTorch Example: DNA Sequence Classification")
    print("=" * 60)

    # Convert to PyTorch tensor
    tokens_tensor = torch.from_numpy(tokens).long()

    # Create dummy labels
    labels = torch.randint(0, 2, (num_samples,))

    # Simple embedding + pooling model
    class DNAClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim=64, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=-1)
            self.fc = nn.Linear(embedding_dim, num_classes)

        def forward(self, x):
            # x shape: (batch, seq_len)
            # Mask padding tokens
            mask = (x != -1).float().unsqueeze(-1)  # (batch, seq_len, 1)

            # Embed and mask
            embedded = self.embedding(x.clamp(min=0))  # (batch, seq_len, emb_dim)
            embedded = embedded * mask

            # Global average pooling (ignoring padding)
            pooled = embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, emb_dim)

            # Classify
            logits = self.fc(pooled)  # (batch, num_classes)
            return logits

    # Create model
    vocab_size = 5 ** k
    model = DNAClassifier(vocab_size)

    # Forward pass
    logits = model(tokens_tensor)

    print(f"Model input shape: {tokens_tensor.shape}")
    print(f"Model output shape: {logits.shape}")
    print(f"Embedding table size: {vocab_size + 1:,} × 64 = {(vocab_size + 1) * 64 * 4 / 1e6:.2f} MB")
    print()

    # For large k, use hash reduction
    print("For larger k values (e.g., k=10), use hash_tokens():")
    k_large = 10
    vocab_large = 5 ** k_large
    num_buckets = 10000

    print(f"k={k_large}, original vocab: {vocab_large:,}")
    print(f"After hashing to {num_buckets:,} buckets:")
    print(f"  Embedding table: {num_buckets * 64 * 4 / 1e6:.2f} MB (vs {vocab_large * 64 * 4 / 1e6:.2f} MB)")
    print()

except ImportError:
    print("PyTorch not installed. Skipping PyTorch example.")
    print("Install with: pip install torch")
    print()

# =============================================================================
# TensorFlow Example
# =============================================================================

try:
    import tensorflow as tf

    print("=" * 60)
    print("TensorFlow Example: DNA Sequence Embedding")
    print("=" * 60)

    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    dataset = dataset.batch(32)

    # Simple embedding model
    vocab_size = 5 ** k
    embedding_dim = 64

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            mask_zero=True,  # Mask padding (-1 becomes 0 after clamp)
            name='kmer_embedding'
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])

    # Build model (need to handle negative padding indices)
    # In practice, you'd add +1 to all tokens or use hash_tokens with positive range
    print("Note: In production, add 1 to all tokens or use hash_tokens to avoid negative indices")
    print(f"Model uses Embedding layer with vocab_size={vocab_size:,}")
    print()

except ImportError:
    print("TensorFlow not installed. Skipping TensorFlow example.")
    print("Install with: pip install tensorflow")
    print()

# =============================================================================
# Data augmentation example
# =============================================================================

print("=" * 60)
print("Data Augmentation: Reverse Complement")
print("=" * 60)

# For DNA, reverse complement is a natural augmentation
# Use batch_tokenize_dna_both to get both orientations

tokens_fwd, tokens_rev = bioenc.batch_tokenize_dna_both(
    buffer[:seq_len * 10],  # First 10 sequences
    lengths[:10],
    k=k, max_len=max_len, pad_value=-1,
    stride=1, alphabet="acgtn",
    enable_padding=True  # Return rectangular arrays
)

print(f"Original sequences: {tokens_fwd.shape}")
print(f"Augmented (reverse complement): {tokens_rev.shape}")
print(f"Total training samples: {tokens_fwd.shape[0] * 2}")
print("\nYou can concatenate these for data augmentation:")
print(f"  X_train = np.vstack([tokens_fwd, tokens_rev])")
print(f"  y_train = np.concatenate([labels, labels])")
print()

# =============================================================================
# Note: Sliding window functionality
# =============================================================================

print("=" * 60)
print("Note: Sliding Windows")
print("=" * 60)

print("For sliding window extraction, you can:")
print("1. Extract windows manually before tokenization")
print("2. Use numpy slicing to create overlapping sequence views")
print("3. Process each window as a separate sequence in the batch")
print()

print("=" * 60)
print("Summary")
print("=" * 60)
print("- Use batch_tokenize_dna_shared() for efficient parallel processing")
print("- Canonical strand mode makes models strand-invariant")
print("- batch_tokenize_dna_both() provides natural data augmentation")
print("- hash_tokens() reduces vocabulary for large k values")
print()
