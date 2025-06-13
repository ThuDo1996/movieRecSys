# 🎬 Movie Recommendation System

## 📊 Project Overview

This project builds a movie recommendation system using the **Netflix 1M dataset**, which contains:

- **1 million ratings**
- **3,900 movies**
- **6,040 users**
- Includes **movie metadata** and **user demographic profiles**

We aim to deliver personalized movie recommendations through deep learning models.

---

## 🔧 Tools & Technologies

- **Python** (Data preprocessing, analysis, modeling)
- **Pandas** (Data manipulation)
- **MySQL** (Data storage and querying)
- **Matplotlib** (Visualization)
- **PyTorch** (Model training)
- **Deep Learning Models**: Graph Neural Network, Contrastive Learning

---

## 🔄 Workflow

1. **Data Preprocessing**  
   - Loaded and cleaned data using `pandas`  
   - Handled missing values and filtered sparse users/movies

2. **Data Storage with MySQL**  
   - Imported cleaned data into a MySQL database  
   - Structured user, movie, and rating tables

3. **Data Querying**  
   - Queried rating and metadata using SQL via `pymysql` connector in Python

4. **Visualization**  
   - Plotted distributions of ratings, movie popularity, and user activity using `matplotlib`

5. **Model Training**  
     - **LightGCN [1]**: Captures high-order connectivity by propagating and aggregating information over the user–item interaction graph.
     - **SGL [2]**: Uses the contrastive learning task to enhance the graph representation learning. Random structural augmentation (edge dropping) is used to generate augmented views.
     --**XSimGCL [3]**: Uses the contrastive learning task to enhance the graph representation learning in which augmented views are generated using noise-based augmentation method.
---

## 📌 Key Highlights

- End-to-end data pipeline from ingestion to modeling
- Integrated SQL and Python for scalable analytics
- Compared performance of NCF and GNN in terms of recommendation accuracy

---

## 📁 Dataset

- Netflix Prize Data (1M ratings):  
  [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

---

## ✅ Experimental Results (Model training)

> Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are used as evaluation metrics

|  Method  |    HR@5  |  NDCG@5  |   HR@10  |  NDCG@10 |
|----------|----------|----------|----------|----------|
| LightGCN | Row1Val2 | Row1Val3 | Row1Val4 | Row1Val5 |
|    SGL   | Row2Val2 | Row2Val3 | Row2Val4 | Row2Val5 |
|  XSimGCL | Row3Val2 | Row3Val3 | Row3Val4 | Row3Val5 |

## References
[1] He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 639-648).
[2] Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021, July). Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval (pp. 726-735).
[3] Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2023). XSimGCL: Towards extremely simple graph contrastive learning for recommendation. IEEE Transactions on Knowledge and Data Engineering, 36(2), 913-926.