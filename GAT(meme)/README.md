# EECE7205-Project-HateMeme
All code for the HateMeme Classification project
=======
# Experiment: Harmful Meme Classification with Graph Neural Networks

## Overview
This experiment aims to evaluate the performance of a Graph Attention Network (GAT) on classifying harmful memes using different feature combinations. The baseline is defined as using only **image embeddings** and **text embeddings**. Additional features, such as captions and VQA (Visual Question Answering) outputs, are incrementally added to observe their impact on classification accuracy. Notably, adding VQA resulted in the best performance.

---

## Methodology

### Data Preparation
1. **Input Features**:
   - **Image Embeddings**: Pre-trained embeddings for meme images.
   - **Text Embeddings**: Embeddings extracted from meme-associated text.
   - **Captions**: Textual descriptions generated from meme images.
   - **VQA Outputs**: Answers generated from visual question answering models.

2. **Graph Construction**:
   - Each meme is represented as a graph with two nodes:
     - Image embedding node
     - Meme text embedding node
   - Edges are fully connected between nodes.

3. **Training and Validation Splits**:
   - Train, validation, and test splits were used to evaluate model performance.

### Model

#### Graph Attention Network (GAT) Architecture
The GAT model used in this experiment is designed to capture relationships between nodes in a graph by assigning attention scores to edges. This allows the model to weigh the importance of neighboring nodes when aggregating features. Key components of the GAT architecture include:

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

The enhanced GAT model in this experiment includes hyperparameter optimizations to handle multimodal inputs effectively.

#### GAT Implementation
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EnhancedGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super(EnhancedGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

- **Training Details**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Loss Function: CrossEntropyLoss
  - Scheduler: StepLR with step size of 10 and gamma 0.5
  - Early stopping patience: 5 epochs
  - Batch size: 32

---

## Results
### Experimental Results
| Features Included               | Test Accuracy | Notes                     |
|---------------------------------|---------------|---------------------------|
| Image + Text (Baseline)         | 0.8500        | Baseline                  |
| Baseline + VQA                  | 0.8729        | Best performance          |
| Baseline + Caption              | 0.8503        | Marginal improvement      |
| Baseline + VQA + Caption        | 0.8531        | Slight improvement        |

### Observations
1. **VQA Contributions**:
   - Adding VQA to the baseline resulted in the best performance (0.8729), indicating its importance in capturing additional contextual information.

2. **Captions as a Feature**:
   - Adding captions provided only a marginal improvement (0.8503), suggesting that captions may introduce some redundancy.

3. **Combination of Caption and VQA**:
   - Adding both captions and VQA together provided only a slight improvement over the baseline (0.8531), highlighting potential redundancy between these features.

---

## Conclusion
- **Key Findings**:
  - Adding VQA outputs to the baseline (image and text embeddings) achieved the best performance.
  - Captions offered limited improvement and may introduce noise or redundancy.
  - Combining captions and VQA does not provide substantial additional benefits over using VQA alone.

- **Future Work**:
  - Investigate how to better integrate captions or use alternative captioning models to improve utility.
  - Further analyze the VQA feature to understand which aspects contribute most to performance.

---

## Reproducibility
### Required Files
- `train_embeddings.pt`: Training data embeddings
- `val_embeddings.pt`: Validation data embeddings
- `test_embeddings.pt`: Test data embeddings
- `output_vqa_answers.csv`: image path, text, image caption, VQA outputs

### Training Script
```python
from torch_geometric.data import Data, DataLoader

# Graph creation function omitted for brevity

train_graphs_no_caption_vqa = create_graphs_without_caption_vqa(train_embeddings)
val_graphs_no_caption_vqa = create_graphs_without_caption_vqa(val_embeddings)
test_graphs_no_caption_vqa = create_graphs_without_caption_vqa(test_embeddings)

batch_size = 32
train_loader_no_caption_vqa = DataLoader(train_graphs_no_caption_vqa, batch_size=batch_size, shuffle=True)
val_loader_no_caption_vqa = DataLoader(val_graphs_no_caption_vqa, batch_size=batch_size, shuffle=False)
test_loader_no_caption_vqa = DataLoader(test_graphs_no_caption_vqa, batch_size=batch_size, shuffle=False)

input_dim_no_caption_vqa = train_graphs_no_caption_vqa[0].x.size(-1)  
model_no_caption_vqa = EnhancedGAT(input_dim_no_caption_vqa, hidden_dim, output_dim).to(device)

# Optimizer, criterion, and training loop are described in the main text.
```

For detailed implementation, refer to the script used in the experiment.