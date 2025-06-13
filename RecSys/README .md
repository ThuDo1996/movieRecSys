### 🔧 Model Training

We implement three graph-based recommendation models: **LightGCN[1]**, **SGL[2]** and **XSimGCL[3]**

#### 📁 Project Structure

- `data_preprocess.py`  
  - Loads the rating dataset  
  - Removes users and items with fewer than 10 ratings 
  - We consider implicit feedback, thus, converting each user-item pair is label as 1. 
  - Splits the data into training, validation, and test sets (70:10:20)

- `data_loader.py`  
   - Load and reprare data

- `LightGCN.py`, `SGL.py` and `XSimGCL.py`  
  - Contain implementations of **LightGCN**, **SGL** and **XSimGCL** models

- `main.py`  
  - Entry point for model training  
  - Handles training, evaluation, and result saving

- `plot.py`  
  - We define top 5% of items (by interaction count) as popular with the rest classified as unpopular. 
  - we sample 1,000 popular items, 1,000 unpopular items, and 2,000 users and visualize their 2D t-SNE projections.

#### 🧠 Model Descriptions

- **LightGCN**  
  A lightweight graph-based model that captures user–item interactions via graph convolution without non-linear activations.

- **SGL** and **XSimGCL**  
  The combination of LightGCN as base model with contrastive learning task to enhance graph representation learning. In **SGL**, augmented views are generated using random structural augmentation (e.g edge dropping) while in **XSimGCL** noise-based augmentation method are used. 

You can switch between models in `main.py` by changing the model selection parameter.

#### Usage
- Run `python data_preprocess.py` to create training, validation, and test sets
- Run `python main.py` to train the model. 

#### Experimental Results 


- **LightGCN** aggregates information uniformly from neighboring nodes, inadventently amplifying the influence of popular nodes due to their high connectivity. This introduce the popularity bias that skews representation learning. It leads to the dominance of popular items in recommendation lists often reduces visibility for unpopular ones, resulting in a performance imbalance that degrades overall effectiveness.
- **SGL** and **XSimGCL** address this issue by incoporating contrastive learning. It enhances the representation learning by pulling similar nodes closer and pushing dissimilar ones apart, thereby improving the generalization. 

#### References:

[1] He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 639-648).
[2] Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021, July). Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval (pp. 726-735).
[3] Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2023). XSimGCL: Towards extremely simple graph contrastive learning for recommendation. IEEE Transactions on Knowledge and Data Engineering, 36(2), 913-926.