import torch
import torch.nn.functional as F

# Model Building
# Automatic Parameter Tuning Tool
#
# Repeat three rounds:
#     Each time, take a small packet of data:
#         1. Predict
#         2. Determine the error rate
#         3. Calculate how to correct it
#         4. Modify the model

torch.manual_seed(123)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
# An automatic parameter tuning tool
# SGD: If the model is wrong, adjust it slightly in a way that makes it "better than expected".
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# Run the entire package of materials 3 times
num_epochs = 3

# Repeat the entire dataset 3 times
for epoch in range(num_epochs):

    model.train()  # Tell the model: "This is training mode."

    # Do not consume all the materials at once.
    # Consume one small packet (batch) at a time.
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        # How many guesses were made incorrectly by the calculation model?
        loss = F.cross_entropy(logits, labels)

        # Clear old gradients
        optimizer.zero_grad()
        # How can the calculation be improved?
        loss.backward()
        # Really update the model
        optimizer.step()

        ### LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.2f}"
        )

    model.eval() # Switch to evaluation mode
    # Insert optional model evaluation code


## Exercise A.3
# How many parameters does the neural network introduced in listing A.9 have?


