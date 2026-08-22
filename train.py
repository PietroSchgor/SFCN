import torch
import matplotlib.pyplot as plt

class EarlyStopping:
    """Ferma il training in anticipo se la validation loss non migliora dopo una certa patience."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # ---- TRAINING PHASE ----
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)[0] # SFCN restituisce una lista, prendiamo l'elemento 0
            outputs = outputs.view(outputs.size(0), -1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # ---- VALIDATION PHASE ----
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)[0]
                outputs = outputs.view(outputs.size(0), -1)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = 100 * correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Controllo Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping innescato all'epoca {epoch+1}! Nessun miglioramento per {patience} epoche.")
            break
            
    # Plot delle curve di loss a fine addestramento
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.title('Curva di Addestramento (Loss)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, train_losses, val_losses
