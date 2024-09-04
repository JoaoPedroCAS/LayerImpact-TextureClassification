# LayerImpact-TextureClassification

Este repositório contém um projeto que visa testar o impacto de diferentes camadas de redes neurais na classificação de texturas. Ao remover camadas da rede a cada iteração, buscamos medir como cada camada contribui para a performance do modelo, possibilitando alcançar resultados semelhantes com menor custo computacional.

## Objetivo

O principal objetivo deste projeto é avaliar a importância das camadas individuais de várias arquiteturas de redes neurais na tarefa de classificação de texturas. Isso permitirá otimizar modelos, mantendo uma alta precisão com menos recursos computacionais.

## Redes Neurais Utilizadas

O projeto testa as seguintes arquiteturas de redes neurais:
1. ConvNext-Tiny
2. ConvNext-Small
3. ConvNext-Base
4. ConvNext-Large
5. ResNet50
6. InceptionV3
7. VGG19
8. DenseNet211
9. EfficientNetB0
10. MobileNetv2

## Estrutura do Projeto

- `Códigos/`: Contém os códigos para cada rede neural.
- `Resultados/`: Resultados das análises e experimentos.
- `Texturas/`: Pasta que contém o dataset de texturas, no caso os datasets utilizados foram o Flickr Material Dataset [FMD](http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip) e o [1200tex](http://scg-turing.ifsc.usp.br/data/bases/LeavesTex1200.zip).
  
## Requisitos

Para executar este projeto, você precisará dos seguintes pacotes Python:

- `numpy`
- `Pillow`
- `scikit-learn`
- `torch`
- `torchvision`
- `matplotlib`

## Metodologia

Para cada rede neural, seguimos os seguintes passos:
1. Treinamento do modelo completo na tarefa de classificação de texturas.
2. Avaliação do desempenho do modelo completo.
3. Remoção incremental de camadas do modelo.
4. Reavaliação do desempenho do modelo após a remoção de cada camada.

## Resultados

Os resultados dos experimentos serão detalhados na pasta `results/`, com gráficos e médias de desempenho para cada rede neural e cada iteração de remoção de camadas.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para discutir melhorias ou corrigir problemas.


**Contato**

Para mais informações, entre em contato através do email: joao.p.c.a.sa@outlook.com
