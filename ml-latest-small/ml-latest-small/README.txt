🎬 Movie Recommender System (Collaborative Filtering)
📌 Overview

This project implements a Collaborative Filtering–based Movie Recommendation System using Matrix Factorization trained with Stochastic Gradient Descent (SGD).

The system learns hidden (latent) preferences of users and movies from historical ratings and predicts how users might rate unseen movies. Based on these predictions, it recommends the top movies for each user.

This project focuses on building the recommender from scratch to deeply understand how real-world recommendation engines work.

📂 Dataset

The project uses the MovieLens dataset, which contains:

ratings.csv – user–movie ratings

movies.csv – movie metadata (title, genres)

Each rating entry includes:

userId

movieId

rating

timestamp

🧠 Core Concepts
🔹 Collaborative Filtering

Recommendations are generated using user behavior, not movie content.

If users have similar rating patterns, they are likely to enjoy similar movies.

🔹 Matrix Factorization

The sparse user–movie rating matrix is decomposed into:

User latent matrix (W)

Movie latent matrix (X)

Each user and movie is represented by a vector of latent factors.

Prediction formula:

rating = global_mean + user_bias + movie_bias + dot(user_vector, movie_vector)

🔹 Stochastic Gradient Descent (SGD)

The model is trained using SGD:

Updates parameters using one rating at a time

Efficient for large and sparse datasets

🔹 Regularization

Regularization is applied to:

Prevent overfitting

Control the magnitude of latent factors and biases

🔄 Project Workflow
Stage 1: Data Loading

Load ratings and movies datasets

Perform basic exploratory analysis

Stage 2: User–Item Matrix

Rows → Users

Columns → Movies

Values → Ratings

Stage 3: Re-indexing

Map userId and movieId to continuous indices

Required for matrix operations

Stage 4: Model Initialization

User latent matrix

Movie latent matrix

User bias vector

Movie bias vector

Stage 5: Training (SGD)

For each rating:

Predict rating

Compute error

Update:

User bias

Movie bias

User latent factors

Movie latent factors

Includes:

L2 regularization

Early stopping

Train–test split

Stage 6: Evaluation

Model performance is evaluated using:

RMSE (Root Mean Squared Error)

Typical result:

Best Test RMSE ≈ 0.88


This means predictions differ from actual ratings by ~0.88 stars on average.

Stage 7: Recommendations

For a given user:

Predict ratings for all movies

Remove already-rated movies

Sort by predicted rating

Recommend top N movies

🛠 Tech Stack

Python

NumPy
Pandas
Matplotlib
Seaborn

▶️ How to Run
1️⃣ Install Dependencies
pip install numpy pandas matplotlib seaborn

2️⃣ Download Dataset

Download the MovieLens dataset and place the following files in the project directory:

ratings.csv
movies.csv

3️⃣ Run the Project
python recommender.py

📊 Sample Output

Prediction Example:

Movie: 300 (2007)
Actual Rating: 3.5
Predicted Rating: 3.83


Top Recommendations:

1. Movie A – Predicted Rating: 4.21
2. Movie B – Predicted Rating: 4.10
3. Movie C – Predicted Rating: 4.05

⭐ Project Highlights

Collaborative filtering built from scratch

Matrix factorization with bias terms

SGD optimization with regularization

Early stopping to prevent overfitting

Realistic evaluation using RMSE

🚀 Future Improvements

Mini-batch SGD

Hyperparameter tuning

Hybrid recommender (collaborative + content-based)

Neural collaborative filtering

Web UI using React or Streamlit
