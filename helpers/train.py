import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def train_model(model, train_dataset, test_dataset, optimizer, criterion, epochs, batch_size, output_folder, device="cpu"):
    """
    Train a neural network model with specified training and testing datasets.

    This function performs a complete training loop, including:
    - Creating DataLoaders for training and testing datasets
    - Moving the model to the specified device (CPU/GPU)
    - Training the model for a specified number of epochs
    - Tracking and logging training and testing metrics
    - Saving the best and last model weights

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be trained
    train_dataset : torch.utils.data.Dataset
        Dataset used for training the model
    test_dataset : torch.utils.data.Dataset
        Dataset used for evaluating the model's performance
    optimizer : torch.optim.Optimizer
        Optimization algorithm for updating model weights
    criterion : torch.nn.Module
        Loss function used to compute the model's performance
    epochs : int
        Number of complete passes through the entire training dataset
    batch_size : int
        Number of samples processed in a single forward/backward pass
    device : str, optional
        Computing device to use for training (default is "cpu")
        Can be "cpu" or "cuda" for GPU training

    Returns:
    --------
    None

    Side Effects:
    -------------
    - Prints training and testing metrics for each epoch
    - Saves the best performing model to "weights/best_model.pth"
    - Saves the final model to "weights/last_model.pth"

    Example:
    --------
    >>> model = MyModel()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> criterion = nn.CrossEntropyLoss()
    >>> train_model(model, train_dataset, test_dataset, optimizer, criterion, epochs=10, batch_size=32)
    """
    # Ensure weights folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to the specified device
    model = model.to(device)

    best_test_accuracy = 0.0  # Initialize best accuracy tracker

    for epoch in range(epochs):
        # Training phase with progress bar
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap train_loader with tqdm for progress tracking
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)", leave=False)
        
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Accuracy': f'{100 * correct / total:.2f}%'
            })

        # Calculate training accuracy and loss
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Evaluate on the test set with progress bar
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        correct = 0
        total = 0

        # Wrap test_loader with tqdm for progress tracking
        test_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} (Testing)", leave=False)

        with torch.no_grad():
            for images, labels in test_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                test_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}', 
                    'Accuracy': f'{100 * correct / total:.2f}%'
                })

        test_loss /= len(test_loader)
        test_accuracy = 100 * correct / total

        # Log the metrics
        print(
            f"Epoch [{epoch+1}/{epochs}]: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
        )

        # Save the best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), "weights/best_model.pth")

    # Save the last model
    torch.save(model.state_dict(), "weights/last_model.pth")
    
    print("Training completed. Best test accuracy: {:.2f}%".format(best_test_accuracy))
