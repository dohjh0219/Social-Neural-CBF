import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cbf.neural_cbf import NeuralCBFNetwork

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    DATA_PATH = "data/cbf_data.npz"
    MODEL_SAVE_PATH = "models/cbf_model.pth"


    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run generate_data.py first.")
        return

    print("Loading Dataset...")
    data = np.load(DATA_PATH)
    X_numpy = data['X'] # (N, 4)
    Y_numpy = data['Y'] # (N, 1)

    # PyTorch Tensor 변환
    X_tensor = torch.FloatTensor(X_numpy)
    Y_tensor = torch.FloatTensor(Y_numpy)

    # DataLoader 생성 (배치 단위로 쪼개서 학습)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model and critic, optimizer
    model = NeuralCBFNetwork()
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # adam으로 optimize

    # Training Loop
    print(f"Start Training for {EPOCHS} epochs...")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            # predict Forward
            prediction = model(batch_x)
            
            # Compute Loss
            loss = criterion(prediction, batch_y)
            
            # weight update (Backward)
            optimizer.zero_grad() # 이전 기울기 초기화
            loss.backward()       # 기울기 계산
            optimizer.step()      # 가중치 수정
            
            epoch_loss += loss.item()
            
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.6f}")

    # saving model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Finished. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()