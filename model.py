#1
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        nn.init.uniform_(self.gate_scores, -0.1, 0.1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNetwork(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[512, 256], num_classes=10):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_size = input_size
        for h_size in hidden_sizes:
            self.layers.append(PrunableLinear(in_size, h_size))
            in_size = h_size
            
        self.out_layer = PrunableLinear(in_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            
        x = self.out_layer(x)
        return x
    
    def get_sparsity_loss(self):
        loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                loss += torch.sum(gates)
        return loss
        
    def get_sparsity_level(self, threshold=1e-2):
        total_weights = 0
        pruned_weights = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores)
                    pruned = (gates < threshold).sum().item()
                    total = gates.numel()
                    
                    pruned_weights += pruned
                    total_weights += total
                    
        return (pruned_weights / total_weights) * 100 if total_weights > 0 else 0.0
        
    def get_all_gate_values(self):
        all_gates = []
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores)
                    all_gates.append(gates.flatten())
        return torch.cat(all_gates)
