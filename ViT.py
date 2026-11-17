import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
        
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )
        
    def forward(self, x):
        res1 = x
        x = self.layer_norm1(x)
        x, _ = self.mha(x, x, x)
        x = x + res1
        
        res2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + res2
        return x

class MLPHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, num_heads, num_layers, mlp_hidden, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_hidden) for _ in range(num_layers)]
        )
        
        self.mlp_head = MLPHead(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.pos_embedding
        x = self.transformer_blocks(x)
        
        cls_token_out = x[:, 0, :]
        out = self.mlp_head(cls_token_out)
        return out

def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = 10
    train_loader, val_loader = get_dataloaders(64)

    model = VisionTransformer(1, 28, 7, 64, 4, 4, 128, 10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        train_acc = 100 * correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_vit.pth")

    return model, val_loader, device

def visualize_predictions(model, val_loader, device):
    model.eval()
    model.load_state_dict(torch.load("checkpoints/best_vit.pth"))
    
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i][0] * 0.5 + 0.5, cmap='gray')
        ax.set_title(f"Pred: {predicted[i]} | True: {labels[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, loader, current_device = train()
    visualize_predictions(trained_model, loader, current_device)
