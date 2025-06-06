import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from datetime import datetime
import logging
from price_action_processor import PriceActionProcessor
from price_action_model import PriceActionTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PriceActionDataset(Dataset):
    """Dataset for price action data."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class PriceActionTrainer:
    def __init__(
        self,
        model: PriceActionTransformer,
        processor: PriceActionProcessor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        
        # Initialize optimizer and loss functions
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        self.direction_criterion = nn.CrossEntropyLoss()
        self.pattern_criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
    def load_data(self, data_dir: str) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare data from JSON files."""
        all_features = []
        all_labels = []
        
        # Process each asset's data
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                logger.info(f"Processing {filename}...")
                
                with open(filepath, 'r') as f:
                    candles = json.load(f)
                
                # Process candles
                processed_candles = self.processor.process_candles(candles)
                features, labels = self.processor.prepare_model_input(processed_candles)
                
                all_features.append(features)
                all_labels.append(labels)
        
        # Combine all data
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Split into train and validation sets (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets and dataloaders
        train_dataset = PriceActionDataset(X_train, y_train)
        val_dataset = PriceActionDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            direction_logits, pattern_logits, confidence, _ = self.model(features)
            
            # Calculate losses
            direction_loss = self.direction_criterion(direction_logits, labels)
            pattern_loss = self.pattern_criterion(pattern_logits, labels)
            
            # Combined loss with confidence weighting
            loss = direction_loss + 0.5 * pattern_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                direction_logits, pattern_logits, confidence, _ = self.model(features)
                
                # Calculate losses
                direction_loss = self.direction_criterion(direction_logits, labels)
                pattern_loss = self.pattern_criterion(pattern_logits, labels)
                loss = direction_loss + 0.5 * pattern_loss
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(direction_logits, dim=-1)
                true_labels = torch.argmax(labels, dim=-1)
                correct_predictions += (predictions == true_labels).sum().item()
                total_predictions += len(labels)
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            logger.info(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Accuracy: {val_accuracy:.2%}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self.save_model('best_model.pth')
                logger.info("Saved new best model")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
    
    def save_model(self, path: str):
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def main():
    # Initialize components
    processor = PriceActionProcessor(window_size=20)
    model = PriceActionTransformer(
        input_dim=24,  # Number of features from processor
        d_model=128,
        n_heads=4,
        n_layers=3
    )
    
    trainer = PriceActionTrainer(model, processor)
    
    # Load data
    train_loader, val_loader = trainer.load_data('processed')
    
    # Train model
    trainer.train(
        train_loader,
        val_loader,
        n_epochs=50,
        early_stopping_patience=10
    )
    
    # Save final model
    trainer.save_model('final_model.pth')
    logger.info("Training completed and model saved")

if __name__ == "__main__":
    main() 