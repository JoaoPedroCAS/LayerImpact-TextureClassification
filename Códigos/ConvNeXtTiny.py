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
import random
import time
# Define the path to your dataset
dataset_path = 'C:/Users/jpedr/OneDrive/Documentos/IFSC/Texturas'

# Load ConvNeXt-Tiny model and remove the classification layer
class ConvNeXtTinyFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtTinyFeatureExtractor, self).__init__()
        self.model = convnext_tiny(weights=None)
        self.features = nn.Sequential(*list(self.model.children())[:-2])  # Remove classification layer
        self.initialize_weights_more_randomly()

    def save_weights_as_png(self, filename):
        layers = []
        weights = []

        # Collect weights from each layer
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Only collect weights, not biases or batchnorm scalars
                layers.append(name)
                weights.append(param.detach().cpu().numpy().flatten())

        # Plot each layer's weights on the same figure
        plt.figure(figsize=(10, 6))
        
        for i, layer_weights in enumerate(weights):
            plt.scatter([i] * len(layer_weights), layer_weights, s=1, label=layers[i])

        plt.title("Weights Across Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Weight Values")
        plt.xticks(ticks=np.arange(len(layers)), labels=layers, rotation=90)
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(filename)
        plt.close()  # Close the plot to avoid displaying it

    def initialize_weights_more_randomly(self, noise_std_range=(0.1, 0.5)):
        a_std_range = (-5.0, 5.0)
        b_std_range = (-5.0, 5.0)
        random.seed(int(time.time()))
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Ensure a <= b for uniform initialization
                a = random.uniform(*a_std_range)
                b = random.uniform(*b_std_range)
                a, b = min(a, b), max(a, b)
                
                # Use Uniform distribution for weights
                nn.init.uniform_(module.weight, a=a, b=b)  # Wider range of weights
                if module.bias is not None:
                    a = random.uniform(*a_std_range)
                    b = random.uniform(*b_std_range)
                    a, b = min(a, b), max(a, b)
                    nn.init.uniform_(module.bias, a=a, b=b)
                
                # Optionally, add noise to the weights to increase randomness
                with torch.no_grad():
                    module.weight.add_(torch.randn_like(module.weight) * random.uniform(*noise_std_range))
                    if module.bias is not None:
                        module.bias.add_(torch.randn_like(module.bias) * random.uniform(*noise_std_range))
            elif isinstance(module, nn.BatchNorm2d):
                a = random.uniform(*a_std_range)
                b = random.uniform(*b_std_range)
                a, b = min(a, b), max(a, b)
                nn.init.uniform_(module.weight, a=a, b=b)  # Slight randomness in BatchNorm
                a = random.uniform(*a_std_range)
                b = random.uniform(*b_std_range)
                a, b = min(a, b), max(a, b)
                nn.init.uniform_(module.bias, a=a, b=b)
        print("Pesos aleatórios atribuidos!")


    def forward(self, x):
        return self.features(x)
    
    def remove_blocks(self):
        if self.number_of_blocks() > 0:
            blocks = list(self.features[-1][-1].children())
            blocks.pop()
            self.features[-1][-1] = nn.Sequential(*blocks)

    def number_of_blocks(self):
        if len(list(self.features[-1].children())) > 0:
            return len(list(self.features[-1][-1]))
        else:
            return 0
    
    def number_of_sequentials(self):
        return len(list(self.features[-1]))
    
    def remover_sequential(self):
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
print("Transform criado!")

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
print("Imagens carregadas!")

# Convert list of tensors to a single tensor
images_tensor = torch.stack(images)
print("Image tensor criado!")

# Extract features using ConvNeXt-Tiny
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Dispositivo selecionado!")
for i in range(0, 100):
    model = ConvNeXtTinyFeatureExtractor().to(device)
    print(f"Modelo {i} criado")
    model.save_weights_as_png(filename = f"C:\\Users\\jpedr\\OneDrive\\Documentos\\IFSC\\ConvNextTinyRandWeights\\{i}_weights_per_layer.png")
    print(f"Distribuição {i} de pesos salva!")
    blocks = model.number_of_blocks()
    layers_removed = 0
    mean_accuracy_scores = []
    mean_f1_scores = []
    mean_recall_scores = []
    mean_precision_scores = []
    print("Iniciando remoção de blocos!")
    while blocks > 0 or model.number_of_sequentials() > 0:
        model.eval()
        features = []
        with torch.no_grad():
            for j in range(len(images_tensor)):
                img = images_tensor[j].unsqueeze(0).to(device)
                feature = model(img).cpu().numpy()
                features.append(feature.flatten())

        features = np.array(features)
        labels = np.array(labels)
        print("Features extraidas")

        # Perform LDA and cross-validation
        lda = LinearDiscriminantAnalysis()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        accuracy_scores = []
        f1_scores = []
        recall_scores = []
        precision_scores = []
        print("Inicializando o LDA com CV")
        # File to save metrics
        with open(f'C:\\Users\\jpedr\\OneDrive\\Documentos\\IFSC\\ConvNextTinyRandWeights\\{i}_metrics.txt', 'a') as f:  # Change 'w' to 'a' to append to the file
            for train_index, test_index in cv.split(features, labels):
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                print("Treino e teste separado!")

                # Fit the model
                lda.fit(X_train, y_train)
                print("Fit do LDA realizado!")

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
            print("Salvando as métricas!")
            # Write average metrics
            f.write(f"Camadas removidas: {layers_removed}\n")
            f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f}\n")
            f.write(f"Average F1-Score: {np.mean(f1_scores):.4f}\n")
            f.write(f"Average Recall: {np.mean(recall_scores):.4f}\n")
            f.write(f"Average Precision: {np.mean(precision_scores):.4f}\n")
            f.write("\n")
            mean_accuracy_scores.append(np.mean(accuracy_scores))
            mean_f1_scores.append(np.mean(f1_scores))
            mean_recall_scores.append(np.mean(recall_scores))
            mean_precision_scores.append(np.mean(precision_scores))

        layers_removed += 1
        model.remove_blocks()
        blocks = model.number_of_blocks()
        if blocks == 0 and model.number_of_sequentials() > 0:
            print(f"Removendo o {model.number_of_sequentials()} Sequential\n")
            model.remover_sequential()
            blocks = model.number_of_blocks()
            if model.number_of_sequentials() == 0 and blocks == 0:
                break
        print("Bloco removido!")

    # Plotting the metrics
    metrics = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
    scores = [mean_accuracy_scores[::-1], mean_f1_scores[::-1], mean_recall_scores[::-1], mean_precision_scores[::-1]]

    num_metrics = len(metrics)

    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 8), sharex=True)

    for k, (ax, metric, score) in enumerate(zip(axs, metrics, scores)):
        ax.plot(score, marker='o', linestyle='-', label=metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Métricas')
        ax.grid(True)
        if k == num_metrics - 1: 
            ax.set_xlabel('Camadas Removidas')

    # Ajusta o layout para que os subplots não se sobreponham
    plt.savefig(f"C:\\Users\\jpedr\\OneDrive\\Documentos\\IFSC\\ConvNextTinyRandWeights\\{i}_GraficoMetricas.png")
    plt.close()  # Close the plot to avoid displaying it
    print("Gráfico de métricas {i} salvo!")