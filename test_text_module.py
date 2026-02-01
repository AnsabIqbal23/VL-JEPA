from TextModule import TextModule

# Initialize
text_encoder = TextModule()

# Test with sample texts
test_texts = ["cat playing", "dog running", "sunset beach"]

# Get vectors
vectors = text_encoder(test_texts)

# Check output
print(f"Input: {test_texts}")
print(f"Output shape: {vectors.shape}")  # Should be [3, 384]
print(f"First vector (first 5 values): {vectors[0, :5]}")