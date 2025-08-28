# Food Lens: Food Ingredient Detection & Recipe Generation from Images

### 📌 Project Overview
This project implements and compares three iterative prototypes for detecting food items, predicting ingredients, and generating recipes from images:

- **Prototype 1** – Basic ingredient detection using EfficientNet-B0  
- **Prototype 2** – Enhanced multi-modal approach with BLIP for title generation  
- **Prototype 3** – Full end-to-end hybrid system combining EfficientNet-B0, BLIP, and T5 for complete recipe instructions  

An advanced AI system that revolutionizes food analysis by detecting food items from images, identifying ingredients, generating recipe titles, and producing step-by-step cooking instructions using state-of-the-art computer vision and NLP techniques.

### 🧠 Objective
- Detect and classify food items from images with high accuracy.  
- Predict ingredients and generate contextual recipe titles and step-by-step instructions.  
- Evaluate and compare prototypes using standard metrics: Accuracy, F1-score, BLEU, ROUGE, and BERTScore.  
- Handle challenges like complex dishes, ingredient overlaps, and natural language generation quality.

### 🧱 Model Architectures

🔹 **Prototype 1 (Basic Detection)**  
EfficientNet-B0 CNN for food classification and initial ingredient extraction.  
Focuses on core vision tasks but lacks generation capabilities.

🔹 **Prototype 2 (Multi-Modal Title Generation)**  
EfficientNet-B0 enhanced with BLIP (ViT + BERT) for cross-modal alignment.  
Adds title generation via contrastive learning but limited to basic outputs.

🔹 **Prototype 3 (Full End-to-End Hybrid)**  
EfficientNet-B0 + BLIP + T5 (Text-to-Text Transformer).  
Combines detection, title generation, and instruction generation with autoregressive decoding; best for comprehensive recipe creation.

### ⚙️ Implementation
- **Framework**: PyTorch  
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)  
- **Multimodal**: BLIP (Vision Transformer + BERT)  
- **Generation Head**: T5 (encoder-decoder with denoising objective)  
- **Data Augmentation**: Resize (224x224), normalization, rotation, flipping, brightness adjustment via torchvision  
- **Loss Functions**:  
  - Detection Loss: Cross-Entropy  
  - Alignment Loss: Contrastive (BLIP)  
  - Generation Loss: Cross-Entropy (T5)  
- **Hyperparameters**: Batch size = 8-32, Learning rate = 0.0001-0.001, 50-150 epochs  
- **Deployment**: Flask/Django backend, React Native mobile frontend  

### 📊 Evaluation Metrics

| **Metric**      | **Description**                                                     |
|-----------------|-------------------------------------------------------------------|
| Accuracy        | Proportion of correctly classified food items and ingredients    |
| F1-Score        | Harmonic mean of precision and recall for multi-label prediction |
| BLEU            | Measures n-gram overlap for generated titles and instructions    |
| ROUGE-L         | Longest common subsequence for evaluating generation quality     |
| BERTScore F1    | Semantic similarity using BERT embeddings for text generation    |

### 📈 Performance Metrics

| **Prototype** | **Accuracy** | **F1-Score** | **BLEU** | **ROUGE-L** |
|---------------|--------------|--------------|----------|-------------|
| Prototype 1   | 0.85         | 0.82         | N/A      | N/A         |
| Prototype 2   | 0.92         | 0.88         | 0.75     | 0.78        |
| Prototype 3   | 0.95         | 0.92         | 0.82     | 0.85        |

### 📊 Visual Analysis & Graphs
- **Training Curves**: Loss convergence across prototypes, with Prototype 3 showing fastest stabilization (loss ~0.4 after 100 epochs).  
- **Metric Curves**: F1 vs. Epochs peaks at 0.92 for Prototype 3; BLEU/ROUGE improvements in generation tasks.  
- **Ingredient Extraction Curve**: 95% accuracy in Prototype 3, highlighting robust multi-label handling.  
- **Title/Recipe Generation Metrics**: BLEU ~0.82 and ROUGE-L ~0.85, with BERTScore F1 ~0.89 for semantic accuracy.  

### 💡 Why Prototype 3 (Full Hybrid)?
- 🧠 BLIP aligns vision-text for better title context, improving over Prototype 2's basic generation.  
- 🎯 T5 enables detailed instructions from ingredients, reducing hallucinations and enhancing usability.  
- 💡 Reduces computational overhead while achieving end-to-end performance >93% accuracy.  
- 📈 Highest metrics and real-world applicability among all prototypes, ideal for mobile/web deployment.  

---

### 🛠️ Installation

Get started with Food Lens by installing the required dependencies:


---

### 📦 Dataset and Preprocessing
- **Sources**: Food-101, Recipe1M+, custom annotations.  
- **Preprocessing**: Resizing, normalization, augmentation. See `/data_prep/` for scripts.

---

### 📱 Mobile Deployment
- **Frontend**: React Native app in `/mobile_app/`.  
- **Backend**: Flask API in Prototype 3.  
- **Steps**: Train models → run backend → integrate via `/predict` endpoint.  
- **Performance**: ~3 seconds latency on mobile devices.

---

### ⚠️ Limitations and Future Work

**Limitations**  
- Dataset bias towards common cuisines.  
- Requires GPU for training.  
- Struggles with multi-item plates.

**Future Work**  
- Fine-tune with diverse datasets.  
- Cloud deployment (AWS/GCP).  
- Add nutrition analysis and allergy detection.

---



