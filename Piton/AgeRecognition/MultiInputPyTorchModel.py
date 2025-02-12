import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Dataset personalizado
class AgeRangeDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.data = []
        self.labels = []
        self.parts = ["barbilla", "frente", "boca", "ceja_derecho",
                      "ceja_izquierdo"]  # Define parts (excluding original)

        # Map ranges to class indices
        self.age_ranges = sorted(os.listdir(dataset_folder))
        self.range_to_idx = {age_range: idx for idx, age_range in enumerate(self.age_ranges)}

        # Collect data
        for age_range in self.age_ranges:
            age_path = os.path.join(dataset_folder, age_range)
            if os.path.isdir(age_path):
                original_folder = os.path.join(age_path, "original")
                if not os.path.exists(original_folder):
                    continue

                for img_file in os.listdir(original_folder):
                    original_path = os.path.join(original_folder, img_file)
                    if not os.path.isfile(original_path):
                        continue

                    # Check if associated parts exist
                    person_data = {"images": {"original": original_path}, "label": self.range_to_idx[age_range]}

                    all_parts_exist = True
                    for part in self.parts:
                        # Generate the expected filename for this part
                        part_filename = f"{os.path.splitext(img_file)[0]}_{part}.jpg"
                        part_path = os.path.join(age_path, part, part_filename)
                        if os.path.exists(part_path):
                            person_data["images"][part] = part_path
                        else:
                            all_parts_exist = False
                            break

                    if all_parts_exist:
                        self.data.append(person_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        person_data = self.data[idx]
        label = person_data["label"]
        images = {}

        for part, path in person_data["images"].items():
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            images[part] = img

        return images, label


# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preparar dataset
dataset_folder = "C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos"
dataset = AgeRangeDataset(dataset_folder, transform=transform)

# Usar solo el 10% del dataset
indices = list(range(len(dataset)))
_, sampled_indices = train_test_split(indices, test_size=0.1, random_state=42)
sampled_dataset = Subset(dataset, sampled_indices)

# Dividir dataset reducido en entrenamiento y validación
train_size = int(0.8 * len(sampled_dataset))
val_size = len(sampled_dataset) - train_size
train_dataset, val_dataset = random_split(sampled_dataset, [train_size, val_size])

# Crear DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Modelo personalizado
class MultiInputAgeRangeModel(nn.Module):
    def __init__(self, num_classes, parts):
        super(MultiInputAgeRangeModel, self).__init__()
        self.parts = parts

        # Crear un encoder ResNet para cada parte
        self.encoders = nn.ModuleDict({
            part: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            for part in parts
        })
        for encoder in self.encoders.values():
            encoder.fc = nn.Identity()  # Quitar la última capa, usar como extractor de características

        # Clasificador final
        self.fc = nn.Linear(len(parts) * 512, num_classes)  # 512 features por ResNet

    def forward(self, images):
        features = []

        for part in self.parts:
            if part in images and images[part] is not None:
                features.append(self.encoders[part](images[part]))
            else:
                # Usar un tensor de ceros para partes faltantes
                features.append(
                    torch.zeros((images[next(iter(images))].size(0), 512), device=images[next(iter(images))].device))

        # Concatenar características y pasar al clasificador final
        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output


# Inicializar modelo
parts = ["original", "barbilla", "frente", "boca", "ceja_derecho", "ceja_izquierdo"]
num_classes = len(os.listdir(dataset_folder))
model = MultiInputAgeRangeModel(num_classes, parts)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definir pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # Barra de progreso para el entrenamiento
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} (Train)", unit="batch") as pbar:
        for images, labels in train_loader:
            labels = labels.to(device)
            for part in images:
                if images[part] is not None:
                    images[part] = images[part].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)

    # Validación
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # Barra de progreso para la validación
    with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} (Val)", unit="batch") as pbar:
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)
                for part in images:
                    if images[part] is not None:
                        images[part] = images[part].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.update(1)

    # Imprimir métricas
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = correct / total * 100
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
