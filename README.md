# 🎬 Movie Recommender System

This project implements a **Movie Recommendation System** from scratch using **Collaborative Filtering** with **Matrix Factorization**, optimized using **Stochastic Gradient Descent (SGD)**. The system learns hidden (latent) preferences of users and movies from historical ratings to predict ratings for unseen movies, aiming to replicate the core logic behind engines like Netflix and Amazon.

## 🚀 Project Overview

* **Technique:** Collaborative Filtering
* **Model:** Matrix Factorization (with Bias Terms)
* **Optimization:** Stochastic Gradient Descent (SGD)
* **Regularization:** L2 Regularization
* **Evaluation Metric:** RMSE
* **Dataset:** MovieLens
* **Core Components:**
    * User Latent Matrix (W)
    * Movie Latent Matrix (X)
    * User & Movie Bias Terms
    * Dot Product Interaction

## 📂 Dataset
We use the **MovieLens** dataset, which consists of:
* `ratings.csv`: User–movie ratings (includes `userId`, `movieId`, `rating`, `timestamp`)
* `movies.csv`: Movie metadata (includes `title`, `genres`)

## 🧠 Core Concepts

### 🔹 1. Collaborative Filtering
Recommendations are generated based on user behavior rather than movie content. If User A and User B rate movies similarly, they are assumed to share similar preferences.

### 🔹 2. Matrix Factorization
The sparse user–movie rating matrix is decomposed into:
* **W:** User latent matrix
* **X:** Movie latent matrix

### 🔹 3. Prediction Formula
$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$$

Where:
* `Global_Mean`: Average of all ratings
* `User_Bias`: Tendency of a user to rate higher/lower
* `Movie_Bias`: Tendency of a movie to receive higher/lower ratings
* `Dot_Product`: Interaction between user and movie latent features

### 🔹 4. SGD Updates
For each rating $ (u, i) $, parameters are updated to minimize the error:
* `Error` = Actual - Predicted
* **Update Rule:** $param \leftarrow param + \alpha \cdot (Error \cdot input - \lambda \cdot param)$

## 🔄 Project Workflow

1.  **Data Loading:** Load ratings/movies and perform exploratory analysis.
2.  **User–Item Matrix:** Construct matrix where Rows=Users, Cols=Movies, Values=Ratings.
3.  **Re-indexing:** Map IDs to continuous indices.
4.  **Model Initialization:** Initialize latent matrices and bias vectors.
5.  **Training (SGD):** Iterate through ratings, predict, compute error, and update parameters.
6.  **Evaluation:** Calculate RMSE (Typical Best Test RMSE ≈ 0.88).
7.  **Recommendation:** Predict ratings for unseen movies and return Top-N.

## 📊 Hyperparameters

| Hyperparameter | Description |
| :--- | :--- |
| `alpha` | Learning rate |
| `lambda` | Regularization strength |
| `K` | Number of latent factors |
| `epochs` | Number of training iterations |

## 🛠 Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn

## ⭐ Project Highlights

* Collaborative filtering built completely **from scratch**
* Matrix factorization with **bias terms**
* **SGD optimization** with L2 regularization
* **Early stopping** implementation to prevent overfitting
