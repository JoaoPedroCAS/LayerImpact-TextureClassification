import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg19
import matplotlib.pyplot as plt

# Classe base para Feature Extractors
class FeatureExtractorBase(nn.Module):
    def __init__(self):
        super(FeatureExtractorBase, self).__init__()

    def remove_last_block(self):
        raise NotImplementedError

    def remove_last_sequential(self):
        raise NotImplementedError

# Feature Extractor para ResNet50
class ResNet50FeatureExtractor(FeatureExtractorBase):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        print("Inicializando ResNet50 sem pesos pré-treinados.")
        # Carrega o modelo ResNet50 sem pesos pré-treinados
        self.model = resnet50(weights=None)
        self.features = nn.Sequential(*list(self.model.children())[:-2])

        # Inicializar pesos aleatórios em toda a rede
        print("Inicializando pesos aleatórios.")
        self._initialize_random_weights()

    def forward(self, x):
        return self.features(x)

    def remove_last_block(self):
        if len(self.features[-1]) > 0:
            print("Removendo o último bloco.")
            self.features[-1] = nn.Sequential(*list(self.features[-1].children())[:-1])

    def remove_last_sequential(self):
        if len(self.features) > 0:
            print("Removendo o último bloco sequencial.")
            self.features = nn.Sequential(*list(self.features.children())[:-1])

    def _initialize_random_weights(self):
        # Inicializar pesos aleatórios em todas as camadas do modelo
        def init_weights(layer):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Before initialization
        print("Before initialization:", self.model.conv1.weight)

        # Initialize weights
        self.model.apply(init_weights)

        # After initialization
        print("After initialization:", self.model.conv1.weight)

# Feature Extractor para VGG19
class VGG19FeatureExtractor(FeatureExtractorBase):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        print("Inicializando VGG19 sem pesos pré-treinados.")
        self.model = vgg19(weights=None)
        self.features = nn.Sequential(*list(self.model.features.children()))

    def forward(self, x):
        return self.features(x)

    def remove_last_block(self):
        if len(self.features) > 0:
            print("Removendo o último bloco.")
            self.features = nn.Sequential(*list(self.features.children())[:-1])

    def remove_last_sequential(self):
        pass  # Para a VGG19, a remoção de blocos é suficiente

# ImageLoader para carregar as imagens e rótulos
class ImageLoader:
    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        self.transform = transform

    def load_images(self):
        print(f"Carregando imagens do dataset localizado em: {self.dataset_path}")
        images, labels = [], []
        for label, subfolder in enumerate(os.listdir(self.dataset_path)):
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    img_path = os.path.join(subfolder_path, filename)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = self.transform(img)
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Erro ao carregar a imagem {img_path}: {e}")
        print(f"Total de imagens carregadas: {len(images)}")
        return images, labels

# Avaliador de Métricas
class MetricEvaluator:
    def __init__(self):
        self.metrics = {'accuracy': [], 'f1': [], 'recall': [], 'precision': []}
        self.layers_removed = 0

    def compute_metrics(self, features, labels):
        print("Computando métricas.")
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

    def evaluate(self, features, labels, model, images_tensor, device):
        print("Iniciando avaliação do modelo.")
        with open('metrics.txt', 'a') as f:
            while len(model.features) > 0:
                print(f"Avaliando com {len(model.features)} camadas restantes.")
                model.eval()
                extracted_features = []

                with torch.no_grad():
                    for img in images_tensor:
                        img = img.unsqueeze(0).to(device)
                        feature = model(img).cpu().numpy()
                        extracted_features.append(feature.flatten())

                features_np = np.array(extracted_features)
                labels_np = np.array(labels)

                acc, f1, rec, prec = self.compute_metrics(features_np, labels_np)

                print(f"Camadas removidas: {self.layers_removed}")
                f.write(f"Camadas removidas: {self.layers_removed}\n")
                f.write(f"Average Accuracy: {acc:.4f}\n")
                f.write(f"Average F1-Score: {f1:.4f}\n")
                f.write(f"Average Recall: {rec:.4f}\n")
                f.write(f"Average Precision: {prec:.4f}\n\n")

                self.metrics['accuracy'].append(acc)
                self.metrics['f1'].append(f1)
                self.metrics['recall'].append(rec)
                self.metrics['precision'].append(prec)

                self.layers_removed += 1

                if len(model.features[-1]) > 0:
                    model.remove_last_block()
                else:
                    model.remove_last_sequential()

        print("Avaliação concluída.")

    def plot_metrics(self):
        print("Plotando métricas.")
        metrics_list = ['accuracy', 'f1', 'recall', 'precision']
        fig, axs = plt.subplots(len(metrics_list), 1, figsize=(10, 8), sharex=True)

        for i, (ax, metric) in enumerate(zip(axs, metrics_list)):
            ax.plot(self.metrics[metric][::-1], marker='o', linestyle='-', label=metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.grid(True)
            if i == len(metrics_list) - 1:
                ax.set_xlabel('Layers Removed')

        plt.tight_layout()
        plt.show()
        print("Plotagem concluída.")

# Main workflow
def main():
    print("Iniciando o fluxo principal.")
    # Definir o caminho para o dataset
    dataset_path = 'C:/Users/jpedr/OneDrive/Documentos/IFSC/Texturas'

    # Definir as transformações para o pré-processamento das imagens
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carregar o dataset
    image_loader = ImageLoader(dataset_path, transform)
    images, labels = image_loader.load_images()
    images_tensor = torch.stack(images)

    # Escolher o modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_choice = input("Escolha o modelo (resnet50 ou vgg19): ").strip().lower()

    if model_choice == 'resnet50':
        model = ResNet50FeatureExtractor().to(device)
    elif model_choice == 'vgg19':
        model = VGG19FeatureExtractor().to(device)
    else:
        raise ValueError("Modelo não reconhecido. Escolha entre 'resnet50' ou 'vgg19'.")

    # Avaliar métricas
    evaluator = MetricEvaluator()
    evaluator.evaluate(None, labels, model, images_tensor, device)
    evaluator.plot_metrics()

if __name__ == "__main__":
    main()