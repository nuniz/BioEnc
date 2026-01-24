"""
TensorFlow integration examples.

This example demonstrates tensor conversion and integration with TensorFlow
tf.data pipelines for training neural networks on biological sequences.

Requirements:
    pip install bioenc[tensorflow]
    # or
    pip install tensorflow
"""

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    print("or: pip install bioenc[tensorflow]")
    exit(1)

import bioenc

# Example 1: Basic NumPy to TensorFlow conversion
print("=" * 60)
print("Example 1: NumPy to TensorFlow conversion")
print("=" * 60)

sequence = np.frombuffer(b"ATGCGTAAATGA", dtype=np.uint8)
tokens = bioenc.tokenize_dna(sequence, k=3)

# Convert to TensorFlow tensor
tensor = bioenc.to_tensorflow(tokens)

print(f"NumPy tokens: {tokens}")
print(f"TensorFlow tensor: {tensor}")
print(f"Tensor dtype: {tensor.dtype}")
print(f"Tensor shape: {tensor.shape}")
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

# Convert to padded TensorFlow tensor
padded_tensor, seq_lengths = bioenc.batch_to_tensorflow(
    tokens_list,
    pad_value=-1
)

print(f"Batch size: {padded_tensor.shape[0]}")
print(f"Max sequence length: {padded_tensor.shape[1]}")
print(f"Padded tensor:\n{padded_tensor}")
print(f"Sequence lengths: {seq_lengths}")
print()

# Example 3: tf.data.Dataset from Python generator
print("=" * 60)
print("Example 3: tf.data.Dataset with generator")
print("=" * 60)


def dna_sequence_generator(sequences, k=3, stride=3):
    """Generator that yields tokenized DNA sequences."""
    for seq_bytes in sequences:
        seq = np.frombuffer(seq_bytes, dtype=np.uint8)
        tokens = bioenc.tokenize_dna(seq, k=k, stride=stride)
        yield tokens


sequences = [
    b"ATGCGTAAATGA",
    b"TTTGGGCCC",
    b"AAACCCGGGTTT",
    b"GCGTACGTA",
]

# Create dataset from generator
dataset = tf.data.Dataset.from_generator(
    lambda: dna_sequence_generator(sequences, k=3),
    output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int64)
)

print("Dataset samples:")
for i, tokens in enumerate(dataset.take(3)):
    print(f"  Sample {i + 1}: shape={tokens.shape}, tokens={tokens.numpy()}")
print()

# Example 4: Padding and batching with tf.data
print("=" * 60)
print("Example 4: tf.data.Dataset with padding and batching")
print("=" * 60)

# Create dataset with padding
dataset_batched = tf.data.Dataset.from_generator(
    lambda: dna_sequence_generator(sequences, k=3),
    output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int64)
).padded_batch(
    batch_size=2,
    padded_shapes=[None],  # Pad to max length in batch
    padding_values=tf.constant(-1, dtype=tf.int64)
)

print("Batched dataset:")
for batch_idx, batch in enumerate(dataset_batched):
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Shape: {batch.shape}")
    print(f"  Content:\n{batch.numpy()}")
print()

# Example 5: All reading frames to TensorFlow
print("=" * 60)
print("Example 5: All reading frames (6-frame analysis)")
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

# Convert to TensorFlow tensor [batch_size, 6, max_len]
frames_tensor = bioenc.frames_to_tensorflow(all_frames, pad_value=-1)

print(f"Frames tensor shape: {frames_tensor.shape}")
print(f"  Dimension 0: batch size ({frames_tensor.shape[0]})")
print(f"  Dimension 1: number of frames ({frames_tensor.shape[1]})")
print(f"  Dimension 2: max sequence length ({frames_tensor.shape[2]})")
print(f"\nSequence 0, Frame 0: {frames_tensor[0, 0]}")
print(f"Sequence 0, Frame 3 (reverse): {frames_tensor[0, 3]}")
print()

# Example 6: Simple Keras model
print("=" * 60)
print("Example 6: Simple Keras model for DNA sequences")
print("=" * 60)


def create_dna_model(vocab_size=125, embedding_dim=64, hidden_dim=128):
    """
    Create a simple Keras model for DNA sequences.

    Args:
        vocab_size: Size of k-mer vocabulary (5^3 = 125 for k=3, ACGTN)
        embedding_dim: Dimension of k-mer embeddings
        hidden_dim: Dimension of LSTM hidden state

    Returns:
        Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True  # Mask padding values
        ),
        tf.keras.layers.LSTM(hidden_dim),
        tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Create model
model = create_dna_model(vocab_size=125, embedding_dim=64, hidden_dim=128)

print("Model summary:")
model.summary()
print()

# Example forward pass
sequences = [b"ATGCGTAAATGA", b"TTTGGGCCC"]
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths_np = np.array([len(s) for s in sequences], dtype=np.int64)

tokens_list = bioenc.batch_tokenize_dna_shared(buffer, lengths_np, k=3)
padded_np = np.full((len(tokens_list), max(len(t) for t in tokens_list)), -1, dtype=np.int64)
for i, tokens in enumerate(tokens_list):
    padded_np[i, :len(tokens)] = tokens

padded_tensor = tf.constant(padded_np, dtype=tf.int64)

print(f"Input shape: {padded_tensor.shape}")
predictions = model(padded_tensor, training=False)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions:\n{predictions.numpy()}")
print()

# Example 7: Complete training pipeline with tf.data
print("=" * 60)
print("Example 7: Complete training pipeline")
print("=" * 60)


def create_training_dataset(sequences, labels, k=3, batch_size=2):
    """
    Create a complete training dataset.

    Args:
        sequences: List of DNA sequences (bytes)
        labels: List of labels (integers)
        k: K-mer size
        batch_size: Batch size

    Returns:
        tf.data.Dataset ready for training
    """

    def generator():
        for seq_bytes, label in zip(sequences, labels):
            seq = np.frombuffer(seq_bytes, dtype=np.uint8)
            tokens = bioenc.tokenize_dna(seq, k=k, stride=3)
            yield tokens, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # Shuffle, pad, and batch
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=([None], []),
        padding_values=(tf.constant(-1, dtype=tf.int64), tf.constant(0, dtype=tf.int32))
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Create training data
train_sequences = [
    b"ATGCGTAAATGA",
    b"TTTGGGCCC",
    b"AAACCCGGGTTT",
    b"GCGTACGTA",
    b"ATGATGATG",
    b"CCCAAATTT",
]

train_labels = [0, 1, 0, 1, 0, 1]  # Binary labels

train_dataset = create_training_dataset(
    train_sequences,
    train_labels,
    k=3,
    batch_size=2
)

print("Training dataset created")
print("Sample batch:")
for batch_tokens, batch_labels in train_dataset.take(1):
    print(f"  Tokens shape: {batch_tokens.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Tokens:\n{batch_tokens.numpy()}")
    print(f"  Labels: {batch_labels.numpy()}")
print()

# Train for a few steps (demonstration only)
print("Training for 2 epochs (demonstration):")
history = model.fit(
    train_dataset,
    epochs=2,
    verbose=1
)
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
print()

# Example 8: Multi-frame model
print("=" * 60)
print("Example 8: Multi-frame model (6-frame analysis)")
print("=" * 60)


def create_multiframe_model(vocab_size=125, embedding_dim=32, hidden_dim=64):
    """
    Create a model that processes all 6 reading frames.

    Args:
        vocab_size: Size of k-mer vocabulary
        embedding_dim: Dimension of k-mer embeddings
        hidden_dim: Dimension of hidden layers

    Returns:
        Keras model
    """
    # Input: [batch_size, 6, seq_len]
    inputs = tf.keras.Input(shape=(6, None), dtype=tf.int64)

    # Embed each frame
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )

    # Process each frame with LSTM
    frame_outputs = []
    for i in range(6):
        frame = inputs[:, i, :]  # [batch_size, seq_len]
        embedded = embedding_layer(frame)  # [batch_size, seq_len, embedding_dim]
        lstm_out = tf.keras.layers.LSTM(hidden_dim)(embedded)  # [batch_size, hidden_dim]
        frame_outputs.append(lstm_out)

    # Concatenate all frame outputs
    concatenated = tf.keras.layers.Concatenate()(frame_outputs)  # [batch_size, 6*hidden_dim]

    # Final classification
    outputs = tf.keras.layers.Dense(2, activation='softmax')(concatenated)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


multiframe_model = create_multiframe_model(vocab_size=125)

print("Multi-frame model created")
print(f"Input shape: (batch_size, 6, seq_len)")
print(f"Output shape: (batch_size, 2)")
print()

# Test with multi-frame input
sequences = [b"ATGCGTAAATGA"]
buffer = np.frombuffer(b"".join(sequences), dtype=np.uint8)
lengths = np.array([len(s) for s in sequences], dtype=np.int64)

all_frames = bioenc.batch_tokenize_dna_all_frames(buffer, lengths, k=3)
frames_tensor = bioenc.frames_to_tensorflow(all_frames, pad_value=-1)

print(f"Input tensor shape: {frames_tensor.shape}")
predictions = multiframe_model(frames_tensor, training=False)
print(f"Predictions: {predictions.numpy()}")
print()

print("=" * 60)
print("Summary of TensorFlow Integration Features")
print("=" * 60)
print("1. to_tensorflow() - NumPy to TensorFlow conversion")
print("2. batch_to_tensorflow() - Batch conversion with padding")
print("3. frames_to_tensorflow() - Multi-frame tensor conversion")
print("4. Compatible with tf.data.Dataset pipelines")
print("5. Support for tf.keras models")
print("6. Automatic padding and batching")
print("7. Ready for training with model.fit()")
print("8. Multi-frame models for 6-frame analysis")
