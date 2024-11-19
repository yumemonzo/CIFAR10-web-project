import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, criterion, optimizer, device, logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
    
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        loss = total_loss / len(train_loader)
        accuracy = correct / len(train_loader.dataset)

        return loss, accuracy
    
    def valid(self, valid_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

        loss = total_loss / len(valid_loader)
        accuracy = correct / len(valid_loader.dataset)

        return loss, accuracy
    
    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
        
        loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        return loss, accuracy
    
    def training(self, train_loader, valid_loader, config):
        writer = SummaryWriter(config.train.save_dir)
        
        lowest_loss = float('inf')
        for epoch in range(config.train.epochs):
            train_loss, train_acc = self.train(train_loader)
            valid_loss, valid_acc = self.valid(valid_loader)

            self.logger.info(f"Epoch {epoch+1}/{config.train.epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            self.logger.info(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

            writer.add_scalars('Loss', {'Train': train_loss, 'Valid': valid_loss}, epoch)
            writer.add_scalars('Accuracy', {'Train': train_acc, 'Valid': valid_acc}, epoch)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(config.train.save_dir, "best_model.pth"))
                self.logger.info(f"New best model saved with Validation Loss: {valid_loss:.4f}")

        writer.close()
