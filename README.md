# Image Similarity Search for Fashion/Clothes

Welcome to the **Image Similarity Search for Fashion/Clothes** repository! This project explores various techniques to find similar fashion/clothing items based on image features. The experiments range from using basic CNN feature extraction to advanced pretrained models like CLIP, BLIP2, DINO, etc. 

We leverage the [Farfetch Listings Dataset](https://www.kaggle.com/datasets/alvations/farfetch-listings/data) from Kaggle, which contains images of different clothing items. Our goal is to compare and contrast the performance of multiple feature extraction methods for efficient and accurate image-based fashion search.

---

## Table of Contents
- [Image Similarity Search for Fashion/Clothes](#image-similarity-search-for-fashionclothes)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Prerequisites \& Installation](#prerequisites--installation)
  - [Experiments](#experiments)
    - [Experiment 1: CNN-Based Models (VGG, ResNet, Xception)](#experiment-1-cnn-based-models-vgg-resnet-xception)
    - [Experiment 2: Autoencoder Feature Extraction](#experiment-2-autoencoder-feature-extraction)
    - [Experiment 3: Color Histogram Features](#experiment-3-color-histogram-features)
    - [Experiment 4: Advanced Pretrained Models (CLIP, BLIP2, DINO, etc.)](#experiment-4-advanced-pretrained-models-clip-blip2-dino-etc)
  - [Results](#results)
  - [Usage](#usage)
  - [Future Work \& Improvements](#future-work--improvements)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

---

## Project Overview

Fashion similarity search is an important component of modern e-commerce and retail applications. It helps users quickly find similar items by comparing visual and contextual cues. In this repository, we explore multiple ways to extract features from images and measure similarity:

- **Deep CNN-based feature extraction**: Leveraging popular architectures (VGG, ResNet, Xception).  
- **Autoencoder-based approach**: Learning a latent representation in an unsupervised fashion.  
- **Color histogram-based approach**: Utilizing OpenCV-based color histograms.  
- **Advanced pretrained transformers**: Employing CLIP, BLIP2, DINO, EfficientNet, ViT, etc.

By comparing the performance across these methods, we aim to highlight their trade-offs in terms of accuracy and computational cost.

---

## Dataset

We use the **[Farfetch Listings Dataset](https://www.kaggle.com/datasets/alvations/farfetch-listings/data)** from Kaggle, which includes images of various fashion items along with basic metadata. Please follow the steps below to acquire the dataset:

1. **Download from Kaggle**:  
   - Navigate to the [Kaggle dataset page](https://www.kaggle.com/datasets/alvations/farfetch-listings/data).  
   - Download the dataset (images and any accompanying metadata).

2. **Organize data locally**:  
   - Extract the dataset into a folder named `data` at the root of this repository.  
   - Ensure images are located in `data/images` (or adjust paths in the code accordingly).

---

## Project Structure

A possible folder structure is as follows (adjust based on your actual structure):

```
.
├── CNN-Fashion-Folder
│   ├── vgg_similarity.ipynb
│   ├── resnet_similarity.ipynb
│   └── xception_similarity.ipynb
├── Autoencoder-Fashion-Folder
│   └── autoencoder_similarity.ipynb
├── ColorHistogram-Folder
│   └── color_histogram_similarity.ipynb
├── AdvancedPretrained-Folder
│   ├── clip_similarity.ipynb
│   ├── blip2_similarity.ipynb
│   ├── dino_similarity.ipynb
│   ├── efficientnet_similarity.ipynb
│   └── vit_similarity.ipynb
├── data
│   └── images
├── README.md
└── requirements.txt
```

- **CNN-Fashion-Folder**: Notebooks for the CNN-based experiments (VGG, ResNet, Xception).  
- **Autoencoder-Fashion-Folder**: Notebook(s) for the autoencoder-based experiment.  
- **ColorHistogram-Folder**: Notebook(s) for the OpenCV-based color histogram experiment.  
- **AdvancedPretrained-Folder**: Notebooks for the advanced pretrained model experiments (CLIP, BLIP2, DINO, etc.).  
- **data**: Contains the downloaded dataset images from Kaggle.  
- **requirements.txt**: Python environment dependencies.

---

## Prerequisites & Installation

1. **Python Version**  
   - Ensure you have Python 3.7+ installed. 

2. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/image-similarity-fashion.git
   cd image-similarity-fashion
   ```

3. **Install Dependencies**  
   - We recommend using a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows: venv\Scripts\activate
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Download the Dataset**  
   - Follow the steps in [Dataset](#dataset) to download and place images in the `data` folder.

---

## Experiments

### Experiment 1: CNN-Based Models (VGG, ResNet, Xception)

- **Approach**:  
  1. **Feature Extraction**: Use each pretrained model (VGG, ResNet, Xception) to extract feature vectors from the images.  
  2. **Similarity Calculation**: Given a query image, extract its features with the same CNN model and compute cosine similarity with the features of the training images.  
  3. **Results**: Achieved a mean precision of around **70%** across all three models, with **VGG** performing slightly better than ResNet and Xception in our tests.  

- **Notebooks**:  
  - [Training Notebook](./CNN-fashion/Tutorial.ipynb)  

### Experiment 2: Autoencoder Feature Extraction

- **Approach**:  
  1. **Architecture**: A custom autoencoder with an encoder and a decoder.  
  2. **Training**: Due to hardware constraints, we trained for **10 epochs** on **128×128** images.  
  3. **Feature Vector**: The encoder’s latent space serves as the feature representation.  
  4. **Similarity**: Compute cosine similarity in this latent space.  
  5. **Observations**: The model performs reasonably well, but not as accurate as the CNN-based approach.  

- **Notebook**:  
  - [Autoencoder Training Notebook](./autoencoder-fashion/Train.ipynb)
  - [Autoencoder Testing Notebook](./autoencoder-fashion/index-and-test.ipynb)

### Experiment 3: Color Histogram Features

- **Approach**:  
  1. **Region-Based Histograms**: Divide images into regions (e.g., top-left, top-right, bottom-left, bottom-right).  
  2. **Masking & Feature Extraction**: Compute color histograms within each region.  
  3. **Concatenation**: Combine region-based histogram features into a single descriptor.  
  4. **Similarity**: Compare these descriptors using a suitable distance metric (e.g., correlation or chi-square).  
  5. **Limitations**: This method primarily focuses on color information. **Items with the same color but different categories** (e.g., shirt vs. t-shirt) might be mistakenly considered similar.  

- **Notebook**:  
  - [Color Histogram Similarity](./fashion-opencv/test.ipynb)

### Experiment 4: Advanced Pretrained Models (CLIP, BLIP2, DINO, etc.)

- **Approach**:  
  1. **Model Selection**: Tried several state-of-the-art models for feature extraction (CLIP, BLIP2, DINO, EfficientNet, ViT).  
  2. **Feature Extraction**: Used each model’s final or penultimate layer embeddings as image features.  
  3. **Similarity**: Computed similarity between query embeddings and database embeddings.  
  4. **Results**:  
     - **DINO** outperformed the other models in terms of precision and visual similarity quality.  
     - CLIP and BLIP2 also showed strong performance, while EfficientNet and ViT were decent but slightly behind DINO in our tests.  

- **Notebooks**:  
  - [Comparison Notebook](./Different-Models/vit_extractor.ipynb)  

---

## Results

1. **Experiment 1 (CNN)**:  
   - Mean precision ~70% (best performing: **VGG**).  
2. **Experiment 2 (Autoencoder)**:  
   - Decent performance but not as accurate as CNN-based methods.  
3. **Experiment 3 (Color Histograms)**:  
   - Good for color matching, but fails to distinguish different item types with the same color.  
4. **Experiment 4 (Advanced Models)**:  
   - **DINO** stands out as the top performer in our tests.  
   - CLIP and BLIP2 also yield strong results.

---

## Usage

1. **Run the Notebooks**  
   - Open Jupyter Notebook or any compatible environment (e.g., VSCode) and navigate to the folder of interest.  
   - Launch the desired `.ipynb` file.  

2. **Perform a Query**  
   - Load a query image through the notebook’s interface or by specifying a file path.  
   - The code will extract the query’s features and compute similarity against the dataset images.  
   - Visualize the top-k most similar images based on the chosen method.

3. **Modifying Hyperparameters**  
   - Most notebooks allow you to tweak parameters (e.g., batch size, number of epochs for the autoencoder, similarity metrics).  
   - Adjust these as needed for experimentation.  

---

## Future Work & Improvements

1. **Fine-Tuning Advanced Models**:  
   - Instead of using pretrained features only, fine-tuning on a fashion-specific dataset could boost performance.  
2. **Hybrid Models**:  
   - Combine color histogram features with deep-learning-based features to incorporate both color and semantic cues.  
3. **Larger Image Resolutions**:  
   - Train autoencoders or advanced models on higher-resolution images if hardware permits.  
4. **Metadata Integration**:  
   - Leverage textual descriptions and tags (if available) for a multimodal similarity search.  

---

## Contributing

Contributions are welcome!  
1. **Fork** the repository.  
2. **Create** a new branch for your feature (`git checkout -b feature-name`).  
3. **Commit** your changes (`git commit -am 'Add a new feature'`).  
4. **Push** to the branch (`git push origin feature-name`).  
5. **Open a Pull Request**.

Please ensure all changes are well-documented and tested.

---

## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Ravi Tiwari** – [LinkedIn](https://www.linkedin.com/in/travi0764/) | [GitHub](https://github.com/travi0764)  
- For any inquiries, please create an issue or reach out via email at `travi0764@gmail.com`.

---

*Thank you for checking out this project! We hope it helps in your pursuit of fashion-related image similarity and retrieval research.*