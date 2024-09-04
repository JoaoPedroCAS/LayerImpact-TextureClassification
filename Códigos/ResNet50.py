import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = 'C:/Users/jpedr/OneDrive/Documentos/IFSC/Texturas'

# Load ResNet-50 model and remove the classification layer
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
    
    def forward(self, x):
        return self.features(x)
    
    def remove_last_block(self):
        if len(self.features[-1]) > 0:
            self.features[-1] = nn.Sequential(*list(self.features[-1].children())[:-1])

    def remove_last_sequential(self):
        if len(self.features) > 0:
            self.features = nn.Sequential(*list(self.features.children())[:-1])

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder):
    images, labels = [], []
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
images_tensor = torch.stack(images)

# Extract features using ResNet-50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50FeatureExtractor().to(device)

metrics = {'accuracy': [], 'f1': [], 'recall': [], 'precision': []}
layers_removed = 0

def compute_metrics(features, labels):
    lda = LinearDiscriminantAnalysis()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    accuracy_scores, f1_scores, recall_scores, precision_scores = [], [], [], []

    for train_index, test_index in cv.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(np.mean([v['f1-score'] for k, v in report.items() if k.isdigit()]))
        recall_scores.append(np.mean([v['recall'] for k, v in report.items() if k.isdigit()]))
        precision_scores.append(np.mean([v['precision'] for k, v in report.items() if k.isdigit()]))

    return np.mean(accuracy_scores), np.mean(f1_scores), np.mean(recall_scores), np.mean(precision_scores)

with open('metrics.txt', 'a') as f:
    while len(model.features) > 0:
        model.eval()
        features = []

        with torch.no_grad():
            for img in images_tensor:
                img = img.unsqueeze(0).to(device)
                feature = model(img).cpu().numpy()
                features.append(feature.flatten())

        features = np.array(features)
        labels_np = np.array(labels)

        acc, f1, rec, prec = compute_metrics(features, labels_np)

        print(f"Camadas removidas: {layers_removed}")
        f.write(f"Camadas removidas: {layers_removed}\n")
        f.write(f"Average Accuracy: {acc:.4f}\n")
        f.write(f"Average F1-Score: {f1:.4f}\n")
        f.write(f"Average Recall: {rec:.4f}\n")
        f.write(f"Average Precision: {prec:.4f}\n\n")

        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['recall'].append(rec)
        metrics['precision'].append(prec)

        layers_removed += 1

        if len(model.features[-1]) > 0:
            model.remove_last_block()
        else:
            model.remove_last_sequential()

# Plotting the metrics
metrics_list = ['accuracy', 'f1', 'recall', 'precision']
fig, axs = plt.subplots(len(metrics_list), 1, figsize=(10, 8), sharex=True)

for i, (ax, metric) in enumerate(zip(axs, metrics_list)):
    ax.plot(metrics[metric][::-1], marker='o', linestyle='-', label=metric.capitalize())
    ax.set_title(metric.capitalize())
    ax.set_ylim(0, 1)
    ax.grid(True)
    if i == len(metrics_list) - 1:
        ax.set_xlabel('Layers Removed')

plt.tight_layout()
plt.show()
