import torch
from loss import cosine_similarity_loss

def train_step(image, text, target, vision_model, text_model, predictor, optimizer):

    optimizer.zero_grad() # Clear previous gradients

    # These 3 lines convert the raw inputs in the embeddings and then in predictions
    img_vector = vision_model(image)  # Get image representation
    text_vector = text_model(text)     # Get text representation
    pred = predictor(img_vector, text_vector)  # Predict target representation

    # Measure how wrong the prediction is
    loss = cosine_similarity_loss(pred, target)

    # Backpropagate the loss (Compute Gradients)
    loss.backward()

    # Update model parameters
    optimizer.step()

    return loss.item()