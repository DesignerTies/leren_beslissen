import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
import torch

f = './merged_omzet_weer_ticket.csv'
df = pd.read_csv(f, usecols=['Datum_uur', 'Omzet', 'Datum', 'Neerslag', "Tmin", "Tmax", "aantal_tickets", "aantal_opgedaagd"])

def clean_neerslag(v: str):
  if (not isinstance(v, str)): return v
  
  return float(v.replace('mm', '').replace(',', '.').strip())

def clean_temp(v: str):
  if (not isinstance(v, str)): return v

  return float(v.replace('°C', '').strip())

df['Neerslag'] = df['Neerslag'].apply(clean_neerslag)
df['Tmin'] = df['Tmin'].apply(clean_temp)
df['Tmax'] = df['Tmax'].apply(clean_temp)

df['hour'] = pd.to_datetime(df['Datum_uur']).dt.hour
df['day_of_the_week'] = pd.to_datetime(df['Datum']).dt.dayofweek
df['month'] = pd.to_datetime(df['Datum']).dt.month
df["is_weekend"] = (df["day_of_the_week"] >= 5) & (df['day_of_the_week'] < 7)

df = df.dropna(subset=['Omzet'])

feature_columns = [
    'hour', 'day_of_the_week', 'month', 
    'is_weekend', 'Neerslag', 'Tmin', 'Tmax', 'aantal_tickets', 'aantal_opgedaagd'
]
categorical_columns = ['day_of_the_week', 'month']
numerical_columns = ['hour', 'Neerslag', 'Tmin', 'Tmax', 'aantal_tickets', 'aantal_opgedaagd']
binary_columns = ['is_weekend']

X = pd.get_dummies(df[feature_columns], columns=categorical_columns, drop_first=True)
y = df['Omzet']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

class RevenuePredictor(nn.Module):
    def __init__(self, input_size):
        super(RevenuePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = RevenuePredictor(input_size=X_train.shape[1])

# Training parameters instellen
batch_size = 16
num_epochs = 1000
learning_rate = 0.0005

# DataLoader maken voor batched training
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, loss functie en optimizer initialiseren
model = RevenuePredictor(input_size=X_train_scaled.shape[1])
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)

# Training loop
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass en optimalisatie
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Evaluatie op de testset
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
    
    # Print voortgang
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Early stopping
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Evaluatie van het finale model
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor)
    test_predictions = model(X_test_tensor)
    
    # Convert predictions terug naar originele schaal
    train_predictions = y_scaler.inverse_transform(train_predictions.numpy())
    test_predictions = y_scaler.inverse_transform(test_predictions.numpy())
    train_actual = y_scaler.inverse_transform(y_train_tensor.numpy())
    test_actual = y_scaler.inverse_transform(y_test_tensor.numpy())
    
    # Bereken metrics
    train_mse = mean_squared_error(train_actual, train_predictions)
    test_mse = mean_squared_error(test_actual, test_predictions)
    train_r2 = r2_score(train_actual, train_predictions)
    test_r2 = r2_score(test_actual, test_predictions)
    
    print(f"\nFinale resultaten:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")