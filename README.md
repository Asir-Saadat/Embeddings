# Feature Embeddings + Clustering

Using a pretrained ResNet50 to extract features from images and cluster them.

## What it does

- Extracts 2048-dim feature vectors from CIFAR-10 images using ResNet50 (pretrained on ImageNet)
- Clusters the features into 10 groups using K-means
- Visualizes the clusters using t-SNE
- Retrieves similar images given a query image using cosine similarity
- Shows a contact sheet of sample images per cluster
- Computes cluster purity to measure how well clusters align with true labels

## Files

- `main_code.py` — entry point, runs everything
- `models/get_resnet.py` — loads pretrained ResNet50 as a feature extractor
- `data/data_loader.py` — loads CIFAR-10 with preprocessing
- `extract_features.py` — passes images through the model to get feature vectors
- `clustering.py` — runs K-means on the features
- `visualize.py` — t-SNE plot
- `retrieval.py` — nearest neighbor image retrieval
- `contact_sheet.py` — grid of sample images per cluster
- `purity.py` — cluster purity metric

## Results

- Cluster purity: ~0.598 (using frozen ImageNet features, no fine-tuning)

## Dataset

Uses CIFAR-10. Download happens automatically when you run the code.
