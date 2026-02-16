🎬 Movie Recommender System
Collaborative Filtering using Matrix Factorization + SGD

This project implements a Movie Recommendation System from scratch using Collaborative Filtering with Matrix Factorization, optimized using Stochastic Gradient Descent (SGD).

The goal is to deeply understand how modern recommendation engines (like those used by platforms such as Netflix and Amazon) actually work under the hood — mathematically and algorithmically.

🚀 Project Overview

Technique: Collaborative Filtering

Model: Matrix Factorization with Bias Terms

Optimization: Stochastic Gradient Descent (SGD)

Regularization: L2 Regularization

Evaluation Metric: RMSE

Dataset: MovieLens

The system learns hidden (latent) preferences of users and movies from historical ratings and predicts ratings for unseen movies.

📂 Dataset

We use the MovieLens dataset, which contains:

ratings.csv → user–movie ratings

movies.csv → movie metadata (title, genres)

Each rating entry includes:

userId

movieId

rating

timestamp

🧠 Core Concepts
1️⃣ Collaborative Filtering

Recommendations are generated using user behavior only — not movie content.

If two users rate movies similarly, they are likely to enjoy similar movies.

2️⃣ Matrix Factorization

The sparse user–movie rating matrix is decomposed into:

W → User latent matrix

X → Movie latent matrix

Each user and each movie is represented by a vector of hidden features.

🔹 Prediction Formula (Text Format)

Predicted_Rating(u, i) =
Global_Mean

User_Bias[u]

Movie_Bias[i]

Dot_Product(User_Vector[u], Movie_Vector[i])

Where:

Global_Mean = average of all ratings

User_Bias[u] = tendency of user u to rate higher/lower

Movie_Bias[i] = tendency of movie i to receive higher/lower ratings

Dot_Product = interaction between user and movie latent features

3️⃣ Error Calculation

Error = Actual_Rating - Predicted_Rating

4️⃣ SGD Updates

For each rating (u, i):

User_Bias[u] += alpha * (Error - lambda * User_Bias[u])

Movie_Bias[i] += alpha * (Error - lambda * Movie_Bias[i])

User_Vector[u] += alpha * (Error * Movie_Vector[i] - lambda * User_Vector[u])

Movie_Vector[i] += alpha * (Error * User_Vector[u] - lambda * Movie_Vector[i])

Where:

alpha → learning rate

lambda → regularization parameter

5️⃣ RMSE (Evaluation Metric)

RMSE = sqrt( average( (Actual_Rating - Predicted_Rating)^2 ) )

Typical result:

Best Test RMSE ≈ 0.88

This means predictions differ from actual ratings by ~0.88 stars on average.

🔄 Project Workflow
Stage 1: Data Loading

Load ratings and movies dataset

Perform basic exploratory analysis

Stage 2: User–Item Matrix

Rows → Users

Columns → Movies

Values → Ratings

Stage 3: Re-indexing

Map userId and movieId to continuous indices for matrix operations.

Stage 4: Model Initialization

Initialize:

User latent matrix

Movie latent matrix

User bias vector

Movie bias vector

Stage 5: Training (SGD)

For each rating:

Predict rating

Compute error

Update parameters

Apply regularization

Includes:

Train–test split

L2 regularization

Early stopping

Stage 6: Evaluation

Example:

Movie: 300 (2007)
Actual Rating: 3.5
Predicted Rating: 3.83

Stage 7: Recommendations

For a given user:

Predict ratings for all movies

Remove already-rated movies

Sort by predicted rating

Recommend Top-N movies

Example:

Movie A – Predicted Rating: 4.21

Movie B – Predicted Rating: 4.10

Movie C – Predicted Rating: 4.05

📊 Hyperparameters
Hyperparameter	Description
alpha	Learning rate
lambda	Regularization strength
K	Number of latent factors
epochs	Number of training iterations
🛠 Tech Stack

Python

NumPy

Pandas

Matplotlib

Seaborn

⭐ Project Highlights

Collaborative filtering built completely from scratch

Matrix factorization with bias terms

SGD optimization with regularization

Early stopping to prevent overfitting

Proper evaluation using RMSE

🚀 Future Improvements

Mini-batch SGD

Hyperparameter tuning

Hybrid recommender (Collaborative + Content-based)

Neural Collaborative Filtering

Web UI using Streamlit or React
