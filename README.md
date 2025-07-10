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
| LightGCN |  0.5827  |  0.4288  |  0.7368  |  0.4788  |
|    SGL   |  0.5881  |  0.4336  |  0.7396  |  0.4828  |

## References
[1] He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (pp. 639-648).

[2] Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021, July). Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval (pp. 726-735).
