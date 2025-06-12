# 🎬 Netflix Movie Recommendation System

## 📊 Project Overview

This project builds a movie recommendation system using the **Netflix 1M dataset**, which contains:

- **1 million ratings**
- **3,900 movies**
- **6,040 users**
- Includes **movie metadata** and **user demographic profiles**

We aim to deliver personalized movie recommendations through deep learning models and graph-based techniques.

---

## 🔧 Tools & Technologies

- **Python** (Data preprocessing, analysis, modeling)
- **Pandas** (Data manipulation)
- **MySQL** (Data storage and querying)
- **Matplotlib** (Visualization)
- **PyTorch** (Model training)
- **Deep Learning Models**: 
  - Neural Collaborative Filtering (**NCF**)
  - Graph Neural Network (**GNN**)

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
   - Trained two recommendation models:
     - **Neural Collaborative Filtering (NCF)**: Captures non-linear interactions with MLP layers
     - **Graph Neural Network (GNN)**: Leverages user-item interaction graph for structural learning

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

## ✅ Results

> Coming soon: Accuracy scores and evaluation metrics across both models

---

## 📎 How to Run

```bash
# Clone the repository
git clone https://github.com/ThuDo1996/movieRecSys
cd movieRecSys

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python preprocess.py

# Train model
python train_ncf.py  # or python train_gnn.py
