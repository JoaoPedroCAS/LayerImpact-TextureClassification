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
import random
import time

# Load ResNet50 model and remove the classification layer
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = resnet50(weights=None)
        # Remove the fully connected layer
        self.features = nn.Sequential(*list(self.model.children())[:-2])  # Remove classification layer
        self.initialize_weights_more_randomly()

    def _get_random_limits(self, a_std_range=(-5.0, 5.0), b_std_range=(-5.0, 5.0)):
        """Gera limites aleatórios para a inicialização uniforme."""
        a = random.uniform(*a_std_range)
        b = random.uniform(*b_std_range)
        return min(a, b), max(a, b)

    def initialize_weights_more_randomly(self, noise_std_range=(0.1, 0.5)):
        """Inicializa pesos e vieses com valores aleatórios e ruído."""
        random.seed(42)  # Use um valor fixo ou configure conforme necessário
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                a, b = self._get_random_limits()
                nn.init.uniform_(module.weight, a=a, b=b)
                if module.bias is not None:
                    a, b = self._get_random_limits()
                    nn.init.uniform_(module.bias, a=a, b=b)
                with torch.no_grad():
                    noise_std = random.uniform(*noise_std_range)
                    module.weight.add_(torch.randn_like(module.weight) * noise_std)
                    if module.bias is not None:
                        module.bias.add_(torch.randn_like(module.bias) * noise_std)
            elif isinstance(module, nn.BatchNorm2d):
                a, b = self._get_random_limits()
                nn.init.uniform_(module.weight, a=a, b=b)
                nn.init.uniform_(module.bias, a=a, b=b)
        print("Pesos aleatórios atribuídos!")

    def save_weights_as_png(self, filename):
        """Salva a distribuição de pesos das camadas em um arquivo PNG."""
        layers, weights = [], []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                layers.append(name)
                weights.append(param.detach().cpu().numpy().flatten())
        plt.figure(figsize=(10, 6))
        for i, layer_weights in enumerate(weights):
            plt.scatter([i] * len(layer_weights), layer_weights, s=1, label=layers[i])
        plt.title("Weights Across Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Weight Values")
        plt.xticks(ticks=np.arange(len(layers)), labels=layers, rotation=90)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def forward(self, x):
        return self.features(x)

    def remove_blocks(self):
        """Remove blocos da última camada."""
        if self.number_of_blocks() > 0:
            blocks = list(self.features[-1][-1].children())
            if blocks:
                blocks.pop()
                self.features[-1][-1] = nn.Sequential(*blocks)

    def number_of_blocks(self):
        """Retorna o número de blocos na última camada."""
        last_layer = self.features[-1]
        if isinstance(last_layer, nn.Sequential) and len(list(last_layer.children())) > 0:
            return len(list(last_layer[-1].children()))
        return 0
    
    def number_of_sequentials(self):
        """Retorna o número de módulos nn.Sequential na última camada."""
        return len(list(self.features[-1]))
    
    def remove_sequential(self):
        """Remove o último módulo nn.Sequential da última camada."""
        if self.number_of_sequentials() > 0:
            sequential = list(self.features[-1].children())
            if sequential:
                sequential.pop()
                self.features[-1] = nn.Sequential(*sequential)

def get_dataset_paths(env):
    """Retorna os caminhos do dataset e de salvamento baseados no ambiente."""
    if env == 1:
        return 'C:/Users/jpedr/OneDrive/Documentos/IFSC/Texturas', 'C:/Users/jpedr/OneDrive/Documentos/IFSC/ConvNextTinyRandWeights'
    return '~/Projetos/LayerImpact-TextureClassification/Texturas', '~/Projetos/LayerImpact-TextureClassification/ConvNextTinyRandWeights'

def ensure_directories(paths):
    """Garante que os diretórios existem."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def load_images_from_folder(folder, transform):
    """Carrega e transforma imagens de um diretório."""
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

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Transform criado!")

# Determine environment and paths
env = int(input("Qual ambiente você está utilizando? (1 - Windows / 2 - Linux)"))
dataset_path, save_results = get_dataset_paths(env)
ensure_directories([dataset_path, save_results])

# Load dataset
images, labels = load_images_from_folder(dataset_path, transform)
print("Imagens carregadas!")

# Convert list of tensors to a single tensor
images_tensor = torch.stack(images)
print("Image tensor criado!")

def evaluate_model(model, images_tensor, labels, device, save_results, iteration):
    """Avalia o modelo e salva as métricas em um arquivo."""
    model.eval()
    features = []
    with torch.no_grad():
        for img in images_tensor:
            img = img.unsqueeze(0).to(device)
            feature = model(img).cpu().numpy()
            features.append(feature.flatten())
    
    features = np.array(features)
    labels = np.array(labels)
    print("Features extraídas")

    lda = LinearDiscriminantAnalysis()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    accuracy_scores, f1_scores, recall_scores, precision_scores = [], [], [], []

    with open(f'{save_results}/{iteration}_metrics.txt', 'a') as f:
        for train_index, test_index in cv.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            print("Treino e teste separados!")

            lda.fit(X_train, y_train)
            print("Fit do LDA realizado!")

            y_pred = lda.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(np.mean([report.get(str(i), {}).get('f1-score', 0) for i in range(len(np.unique(labels)))]))
            recall_scores.append(np.mean([report.get(str(i), {}).get('recall', 0) for i in range(len(np.unique(labels)))]))
            precision_scores.append(np.mean([report.get(str(i), {}).get('precision', 0) for i in range(len(np.unique(labels)))]))
        
        print("Salvando as métricas!")
        f.write(f"Camadas removidas: {iteration}\n")
        f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f}\n")
        f.write(f"Average F1-Score: {np.mean(f1_scores):.4f}\n")
        f.write(f"Average Recall: {np.mean(recall_scores):.4f}\n")
        f.write(f"Average Precision: {np.mean(precision_scores):.4f}\n")
        f.write("\n")

    return np.mean(accuracy_scores), np.mean(f1_scores), np.mean(recall_scores), np.mean(precision_scores)

# Loop principal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Dispositivo selecionado!")

for i in range(100):
    model = ResNet50FeatureExtractor().to(device)
    print(f"Modelo {i} criado")
    model.save_weights_as_png(filename=f"{save_results}/{i}_weights_per_layer.png")
    print(f"Distribuição {i} de pesos salva!")

    mean_accuracy_scores, mean_f1_scores, mean_recall_scores, mean_precision_scores = [], [], [], []
    blocks = model.number_of_blocks()
    layers_removed = 0
    print("Iniciando remoção de blocos!")

    while blocks > 0 or model.number_of_sequentials() > 0:
        acc, f1, recall, precision = evaluate_model(model, images_tensor, labels, device, save_results, i)
        mean_accuracy_scores.append(acc)
        mean_f1_scores.append(f1)
        mean_recall_scores.append(recall)
        mean_precision_scores.append(precision)

        layers_removed += 1
        model.remove_blocks()
        blocks = model.number_of_blocks()
        if blocks == 0 and model.number_of_sequentials() > 0:
            print(f"Removendo o {model.number_of_sequentials()} Sequential\n")
            model.remove_sequential()
            blocks = model.number_of_blocks()
            if model.number_of_sequentials() == 0 and blocks == 0:
                break
        print("Bloco removido!")

    # Plotting the metrics
    metrics = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
    scores = [mean_accuracy_scores[::-1], mean_f1_scores[::-1], mean_recall_scores[::-1], mean_precision_scores[::-1]]

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 8), sharex=True)
    for k, (ax, metric, score) in enumerate(zip(axs, metrics, scores)):
        ax.plot(score, marker='o', linestyle='-', label=metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Métricas')
        ax.grid(True)
        if k == len(metrics) - 1:
            ax.set_xlabel('Camadas Removidas')

    plt.tight_layout()
    plt.savefig(f"{save_results}/{i}_GraficoMetricas.png")
    plt.close()
    print(f"Gráfico de métricas {i} salvo!")
