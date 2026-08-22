import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# Assumiamo che la repo SFCN sia aggiunta al sys.path in Kaggle
from dp_model import dp_loss as dpl

def train_model(model, train_loader, val_loader, optimizer, device, epochs=130):
    # Scheduler: LR moltiplicato per 0.3 ogni 2855 epoche (calcolato per eguagliare gli step del paper originale con 170 pazienti)
    scheduler = StepLR(optimizer, step_size=2855, gamma=0.3)
    
    train_losses = []
    val_losses = []
    val_maes = []
    
    best_mae = float('inf')
    best_model_state = None
    
    # Range di binning che useremo (0-70 anni, con step di 1)
    bin_centers = np.arange(0, 70, 1)
    
    for epoch in range(epochs):
        # ---- TRAINING PHASE ----
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels_vect, _) in enumerate(train_loader):
            inputs, labels_vect = inputs.to(device), labels_vect.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)[0] # SFCN restituisce una lista
            outputs = outputs.view(outputs.size(0), -1)
            
            # Loss: KL Divergence (soft-classification)
            loss = dpl.my_KLDivLoss(outputs, labels_vect)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # ---- VALIDATION PHASE ----
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for inputs, labels_vect, true_age in val_loader:
                inputs, labels_vect = inputs.to(device), labels_vect.to(device)
                true_age = true_age.numpy()
                
                outputs = model(inputs)[0]
                outputs = outputs.view(outputs.size(0), -1)
                
                loss = dpl.my_KLDivLoss(outputs, labels_vect)
                val_loss += loss.item() * inputs.size(0)
                
                # Calcolo dell'età predetta e del MAE (Mean Absolute Error)
                prob = torch.exp(outputs).cpu().numpy()
                predicted_age = prob @ bin_centers
                
                mae = np.sum(np.abs(predicted_age - true_age))
                val_mae += mae
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} anni")
        
        # L'epoca con il miglior validation MAE è usata per il test finale (SFCN paper)
        if val_mae < best_mae:
            best_mae = val_mae
            best_model_state = model.state_dict().copy()
            print(f"  -> Nuovo miglior MAE: {best_mae:.2f}! Modello salvato in cache.")
            
        # Step dello scheduler
        scheduler.step()
            
    print(f"\\nTraining completato! Carico i pesi dell'epoca con il miglior MAE in validazione ({best_mae:.2f}).")
    model.load_state_dict(best_model_state)
            
    # Plot delle curve a fine addestramento
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence Loss')
    plt.legend()
    plt.title('Curva di Loss (Soft-Classification)')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_maes, label='Validation MAE', color='red', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('MAE (anni)')
    plt.legend()
    plt.title('Mean Absolute Error')
    
    plt.tight_layout()
    plt.show()
    
    return model, train_losses, val_losses, val_maes
