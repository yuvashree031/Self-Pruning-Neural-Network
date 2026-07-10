import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import os

from model import SelfPruningNetwork


# train and evaluate
def train_and_evaluate(lam, epochs=5, device='cpu'):
    print(f"\n{'='*40}")
    print(f" Training with Lambda: {lam}")
    print(f"{'='*40}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                              shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                             shuffle=False, num_workers=0)
    
    model = SelfPruningNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    
    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 0.001},
        {'params': gate_params, 'lr': 0.05}
    ])
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_sparsity_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            cls_loss = criterion(outputs, labels)
            
            sparsity_loss = model.get_sparsity_loss()
            
            total_loss = cls_loss + lam * sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_sparsity_loss += sparsity_loss.item()
            
        avg_loss = running_loss / len(trainloader)
        avg_cls_loss = running_cls_loss / len(trainloader)
        avg_sp_loss = running_sparsity_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.4f} "
              f"(Cls: {avg_cls_loss:.4f}, Sparsity: {avg_sp_loss:.4f})")
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    sparsity = model.get_sparsity_level()
    
    print(f"\n--- Final Results for Lambda={lam} ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")
    
    return accuracy, sparsity, model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lambdas = [0.001, 0.01, 0.1]
    results = []
    
    best_model = None
    best_lambda_to_plot = 0.1
    
    for lam in lambdas:
        acc, sparsity, model = train_and_evaluate(lam, epochs=5, device=device)
        results.append({
            "lambda": lam,
            "accuracy": acc,
            "sparsity": sparsity
        })
        
        if lam == best_lambda_to_plot:
            best_model = model
            
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    if best_model is not None:
        gate_values = best_model.get_all_gate_values().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(gate_values, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        plt.title('Distribution of Final Gate Values (Lambda = 0.1)')
        plt.xlabel('Gate Value (Post-Sigmoid)')
        plt.ylabel('Frequency (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plot_path = 'gate_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved gate distribution plot to {plot_path}")
        
    print("Training process complete!")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
