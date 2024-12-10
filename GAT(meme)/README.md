# EECE7205-Project-HateMeme
All code for the HateMeme Classification project excluding issue and rgcl.

---

## Experiment: Harmful Meme Classification with Graph Neural Networks

### Overview
This experiment evaluates the performance of a Graph Attention Network (GAT) for classifying harmful memes using various feature combinations. The baseline includes **image embeddings** and **text embeddings**. Additional features, such as captions and VQA (Visual Question Answering) outputs, are incrementally added to observe their impact on classification accuracy. Notably, adding VQA resulted in the best performance.

---

## Methodology

### Data Preparation
1. **Input Features**:
   - **Image Embeddings**: Pre-trained embeddings for meme images.
   - **Text Embeddings**: Extracted embeddings from meme-associated text.
   - **Captions**: Textual descriptions generated from meme images.
   - **VQA Outputs**: Answers generated from visual question answering models.

2. **Graph Construction**:
   - Each meme is represented as a graph with two nodes:
     - Image embedding node.
     - Meme text embedding node.
   - Edges are fully connected or constructed using cosine similarity.

3. **Training and Validation Splits**:
   - Train, validation, and test splits were used to evaluate model performance.

---

### Model

#### Graph Attention Network (GAT) Architecture
The GAT model is designed to capture relationships between nodes in a graph by assigning attention scores to edges. This allows the model to weigh the importance of neighboring nodes when aggregating features. Key components of the GAT architecture include:

1. **Input Layer**:
   - Takes node features as input (e.g., image and text embeddings).

2. **Graph Attention Layers**:
   - Applies self-attention mechanisms to compute attention scores for each edge.
   - Aggregates features from neighboring nodes based on the computed attention scores.

3. **Hidden Layers**:
   - Stacks multiple attention layers to learn higher-order feature representations.

4. **Output Layer**:
   - Produces class probabilities for each graph (harmful or non-harmful).

5. **Activation Functions**:
   - Uses non-linear activation functions (e.g., LeakyReLU) to introduce non-linearity.

6. **Dropout**:
   - Regularizes the model by randomly dropping nodes or edges during training to prevent overfitting.

---

### Experimental Results and Comparisons

#### Table II: Performance Evaluation of ResNet and DistilBERT Models with Captions and VQA

| Method         | Caption | VQA | Accuracy (%) | AUROC (%) |
|----------------|---------|-----|--------------|-----------|
| ResNet         |         |     | 61.58        | 59.83     |
| DistilBERT     |         |     | 77.46        | 82.92     |
|                | ✓       |     | 80.79        | 82.40     |
|                |         | ✓   | 80.79        | 82.19     |
|                | ✓       | ✓   | 79.66        | 80.43     |

- **Observations**:
  - Adding captions to DistilBERT slightly improves AUROC, suggesting captions may provide minimal contextual value.
  - VQA inclusion shows consistent performance, but combining both captions and VQA results in a slight degradation.

---

#### Table III: Comparison of Graph Construction Methods (Cosine Similarity vs. Fully Connected) and Pooling Strategies

| Method                            | Accuracy (%) | AUROC (%) |
|-----------------------------------|--------------|-----------|
| Cosine Similarity + Mean Pooling  | 81.41        | 85.85     |
| Cosine Similarity + Attention Pooling | 80.28        | 82.74     |
| Fully Connected + Mean Pooling    | 81.36        | 88.95     |
| Fully Connected + Attention Pooling | 79.66        | 88.05     |

- **Observations**:
  - Fully connected graph structures with mean pooling provide the highest AUROC (88.95%), making it the most robust method for this dataset.
  - Cosine similarity-based graph construction performs comparably in accuracy but with lower AUROC.

---

#### Table IV: Performance Evaluation of Combined ResNet and DistilBERT Models, and the CLIP Model

| Method              | Caption | VQA | Accuracy (%) | AUROC (%) |
|---------------------|---------|-----|--------------|-----------|
| ResNet + DistilBERT |         |     | 75.14        | 78.60     |
|                     | ✓       |     | 75.14        | 80.07     |
|                     |         | ✓   | 77.97        | 78.30     |
|                     | ✓       | ✓   | 76.65        | 79.69     |
| CLIP                |         |     | 83.33        | 89.28     |
|                     | ✓       |     | 83.62        | 90.96     |
|                     |         | ✓   | 84.75        | 90.94     |
|                     | ✓       | ✓   | 81.36        | 88.95     |

- **Observations**:
  - The CLIP model demonstrates the best performance when combining VQA with captions, achieving an AUROC of 90.96%.
  - ResNet + DistilBERT underperforms compared to CLIP, but adding captions improves its AUROC significantly.

---

### Revised Findings

1. **Graph Construction Methods**:
   - Fully connected graph structures with mean pooling are most effective for capturing relationships between nodes.
   - Cosine similarity-based construction performs well for accuracy but slightly lags in AUROC.

2. **Feature Importance**:
   - VQA and captions contribute significantly to performance, particularly when combined with the CLIP model.
   - DistilBERT and ResNet models show modest improvements with captions and VQA but are still outperformed by CLIP.

3. **CLIP's Robustness**:
   - The CLIP model consistently achieves the highest AUROC scores across scenarios, highlighting its capability to capture multimodal relationships.