import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = 'C:/Users/jpedr/OneDrive/Documentos/IFSC/Texturas'

# Load ConvNeXt-Tiny model and remove the classification layer
class ConvNeXtTinyFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtTinyFeatureExtractor, self).__init__()
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.model.children())[:-2])  # Remove classification layer

    def forward(self, x):
        return self.features(x)

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
model = ConvNeXtTinyFeatureExtractor().to(device)
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
with open('metrics.txt', 'w') as f:
    for train_index, test_index in cv.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Fit the model
        lda.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = lda.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = np.mean([report[str(i)]['f1-score'] for i in range(len(np.unique(labels)))])
        recall = np.mean([report[str(i)]['recall'] for i in range(len(np.unique(labels)))])
        precision = np.mean([report[str(i)]['precision'] for i in range(len(np.unique(labels)))])

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)

        # Save metrics to file
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write("-" * 20 + "\n")

    # Write average metrics
    f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f}\n")
    f.write(f"Average F1-Score: {np.mean(f1_scores):.4f}\n")
    f.write(f"Average Recall: {np.mean(recall_scores):.4f}\n")
    f.write(f"Average Precision: {np.mean(precision_scores):.4f}\n")

# Plotting the metrics
metrics = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
avg_scores = [
    np.mean(accuracy_scores),
    np.mean(f1_scores),
    np.mean(recall_scores),
    np.mean(precision_scores),
]

plt.bar(metrics, avg_scores)
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()
