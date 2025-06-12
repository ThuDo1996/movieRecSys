### 🔧 Model Training

We implement two deep learning models for recommendation: **LightGCN [1]** and **NeuCF [2]** to making recommendations.

#### 📁 Project Structure

- `data_preprocess.py`  
  - Loads the rating dataset  
  - Removes users and items with fewer than 10 ratings 
  - We consider implicit feedback, thus, converting each user-item pair is label as 1. 
  - Splits the data into training, validation, and test sets (70:10:20)

- `data_loader.py`  
   - Load and reprare data

- `LightGCN.py` and `NeuCF.py`  
  - Contain implementations of **LightGCN** and **Neural Collaborative Filtering (NeuCF)** models

- `main.py`  
  - Entry point for model training  
  - Handles training, evaluation, and result saving

#### 🧠 Model Descriptions

- **LightGCN**  
  A lightweight graph-based model that captures user–item interactions via graph convolution without non-linear activations.

- **NeuCF**  
  A hybrid neural model combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) to learn complex interaction patterns.

You can switch between models in `main.py` by changing the model selection parameter.

#### Usage
- Run `python data_preprocess.py` to create training, validation, and test sets
- Run `python main.py` to train the model. 
#### References:
[1]He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 639-648).

[2]He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).
