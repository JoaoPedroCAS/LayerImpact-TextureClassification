import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = 'C:/Users/jpedr/OneDrive/Documentos/IFSC/Texturas'

# Load ConvNeXt-Tiny model and remove the classification layer
class ConvNeXtSmallFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtSmallFeatureExtractor, self).__init__()
        self.model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.model.children())[:-2])  # Remove classification layer

    def forward(self, x):
        return self.features(x)
    
    def remove_blocks(self):
        if self.number_of_blocks() > 0:
            blocos = list(self.features[-1][-1].children())
            blocos.pop()
            self.features[-1][-1] = nn.Sequential(*blocos)

    def number_of_blocks(self):
        if len(list(self.features[-1].children())) > 0:
            return len(list(self.features[-1][-1]))
        else:
            return 0
    
    def number_of_sequentials(self):
        return len(list(self.features[-1]))
    
    def remove_sequential(self):
        if self.number_of_sequentials() > 0:
            sequential = list(self.features[-1].children())
            sequential.pop()
            self.features[-1] = torch.nn.Sequential(*sequential)

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return images, labels

# Load dataset
images, labels = load_images_from_folder(dataset_path)

# Convert list of tensors to a single tensor
images_tensor = torch.stack(images)

# Extract features using ConvNeXt-Tiny
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNeXtSmallFeatureExtractor().to(device)
blocos = model.number_of_blocks()
camadas_removidas = 0
mean_accuracy_scores = []
mean_f1_scores = []
mean_recall_scores = []
mean_precision_scores = []
while blocos > 0 or model.number_of_sequentials() > 0:
    model.eval()

    features = []
    with torch.no_grad():
        for i in range(len(images_tensor)):
            img = images_tensor[i].unsqueeze(0).to(device)
            feature = model(img).cpu().numpy()
            features.append(feature.flatten())

    features = np.array(features)
    labels = np.array(labels)

    # Perform LDA and cross-validation
    lda = LinearDiscriminantAnalysis()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

    # File to save metrics
    with open('metrics.txt', 'a') as f:  # Change 'w' to 'a' to append to the file
        for train_index, test_index in cv.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Fit the model
            lda.fit(X_train, y_train)

            # Predict and evaluate the model
            y_pred = lda.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = np.mean([report[str(i)]['f1-score'] for i in range(len(np.unique(labels))) if str(i) in report])
            recall = np.mean([report[str(i)]['recall'] for i in range(len(np.unique(labels))) if str(i) in report])
            precision = np.mean([report[str(i)]['precision'] for i in range(len(np.unique(labels))) if str(i) in report])

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
            recall_scores.append(recall)
            precision_scores.append(precision)

        # Write average metrics
        print(f"Camadas removidas: {camadas_removidas}\n")
        f.write(f"Camadas removidas: {camadas_removidas}\n")
        f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f}\n")
        f.write(f"Average F1-Score: {np.mean(f1_scores):.4f}\n")
        f.write(f"Average Recall: {np.mean(recall_scores):.4f}\n")
        f.write(f"Average Precision: {np.mean(precision_scores):.4f}\n")
        f.write("\n")
        mean_accuracy_scores.append(np.mean(accuracy_scores))
        mean_f1_scores.append(np.mean(f1_scores))
        mean_recall_scores.append(np.mean(recall_scores))
        mean_precision_scores.append(np.mean(precision_scores))

    camadas_removidas += 1
    model.remove_blocks()
    blocos = model.number_of_blocks()
    if blocos == 0 and model.number_of_sequentials() > 0:
        print(f"Removendo o {model.number_of_sequentials()} Sequential\n")
        model.remove_sequential()
        blocos = model.number_of_blocks()
        if model.number_of_sequentials() == 0 and blocos == 0:
            break

# Plotting the metrics
metrics = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
scores = [mean_accuracy_scores[::-1], mean_f1_scores[::-1], mean_recall_scores[::-1], mean_precision_scores[::-1]]

num_metrics = len(metrics)

fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 8), sharex=True)

for i, (ax, metric, score) in enumerate(zip(axs, metrics, scores)):
    ax.plot(score, marker='o', linestyle='-', label=metric)
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Métricas')
    ax.grid(True)
    if i == num_metrics - 1: 
        ax.set_xlabel('Camadas Removidas')

# Ajusta o layout para que os subplots não se sobreponham
plt.tight_layout()
plt.show()