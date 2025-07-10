### 🔧 Model Training

We implement three graph-based recommendation models: **LightGCN[1]**, **SGL[2]** and **XSimGCL[3]**

#### 📁 Project Structure

- `data_preprocess.py`  
  - Loads the rating dataset  
  - Removes users and items with fewer than 20 ratings 
  - We consider implicit feedback, thus, converting each user-item pair is label as 1. 
  - Splits the data into training, validation, and test sets (80:10:10)

- `data_loader.py`  
   - Load and reprare data

- `LightGCN.py`, `SGL.py`
  - Contain implementations of **LightGCN**, and **SGL** models

- `main.py`  


#### 🧠 Model Descriptions

- **LightGCN**  
  A lightweight graph-based model that captures user–item interactions via graph convolution without non-linear activations.

- **SGL** 
  The combination of LightGCN as base model with contrastive learning task to enhance graph representation learning. In **SGL**, augmented views are generated using random structural augmentation (e.g edge dropping). 

You can switch between models in `main.py` by changing the model selection parameter.

#### Usage
- Run `python data_preprocess.py` to create training, validation, and test sets
- Run `python main.py` to train the model. 

#### Experimental Results 
> Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are used as evaluation metrics

|  Method  |    HR@5  |  NDCG@5  |   HR@10  |  NDCG@10 |
|----------|----------|----------|----------|----------|
| LightGCN |  0.5827  |  0.4288  |  0.7368  |  0.4788  |
|    SGL   |  0.5881  |  0.4336  |  0.7396  |  0.4828  |

- **LightGCN** aggregates information uniformly from neighboring nodes, which inadvertently amplifies the influence of popular nodes due to their high connectivity. This leads to ***popularity bias***, skewing the representation learning process. As a result, popular items dominate the recommendation lists, reducing the visibility of less popular items and causing performance imbalance that negatively impacts overall effectiveness.
- **SGL** mitigate this issue by incorporating ***contrastive learning***, which enhances representation learning by pulling similar nodes closer while pushing dissimilar ones apart, thus improving generalization. 

#### References:

[1] He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 639-648).

[2] Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021, July). Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval (pp. 726-735).
