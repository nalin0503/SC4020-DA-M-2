from transformers import BertTokenizer, BertModel
import torch

# Load pretrained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# List of phrases
phrases = ["Hello, how are you?", "BERT is a powerful model.", "Transformers are amazing for NLP tasks."]

# Tokenize and process all phrases in the list
inputs = tokenizer(phrases, return_tensors="pt", padding=True, truncation=True)

# Generate embeddings
with torch.no_grad():  # No gradients needed for inference
    outputs = model(**inputs)

# Extract the last hidden states
last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

# Compute phrase embeddings (e.g., mean pooling over tokens for each phrase)
# Extract the `[CLS]` token embedding
phrase_embeddings = last_hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)
  # Shape: (batch_size, hidden_size)

# Convert to numpy matrix for further processing (optional)
embedding_matrix = phrase_embeddings

print("Embedding Matrix Shape:", embedding_matrix.shape)
