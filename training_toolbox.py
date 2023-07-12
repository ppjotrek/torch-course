'''

Basic NN training toolbox to be used in scripts. 

Contains:

- train_step() - Train step (one step of training loop)
- val_step - Validation step (one step of validation loop)
- train_model() - Training loop

'''

import torch

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = None):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_loss, train_acc

def val_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = None):
    val_loss, val_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            val_pred = model(X)
            
            # 2. Calculate loss and accuracy
            val_loss += loss_fn(val_pred, y)
            val_acc += accuracy_fn(y_true=y,
                y_pred=val_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        val_loss /= len(data_loader)
        val_acc /= len(data_loader)
        print(f"val loss: {val_loss:.5f} | val accuracy: {val_acc:.2f}%\n")
        return val_loss, val_acc

#Training loop

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                epochs: int,
                device: torch.device = None):
    
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    
    from training_toolbox import train_step, val_step
    
    for epoch in range(epochs):

        print(f"Epoch: {epoch}\n---------")

        train_loss, train_acc = train_step(data_loader=train_loader, 
            model = model, 
            loss_fn = loss_fn,
            optimizer = optimizer,
            accuracy_fn = accuracy_fn,
            device = device
        )
        val_loss, val_acc = val_step(data_loader=val_loader,
            model = model,
            loss_fn = loss_fn,
            accuracy_fn = accuracy_fn,
            device = device
        )

        print("Epoch complete!\n")
    print("Training complete!")
    print(f"Final metrics: Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}%\n")
    print("Training algorithm by ppjotrek\n")
