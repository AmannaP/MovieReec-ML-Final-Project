🎬 MovieRec: Intelligent Hybrid Recommendation Engine
MovieRec is a production-grade recommendation system developed for the Group 8 Machine Learning Final Assessment. It leverages a Hybrid Architecture combining Collaborative Filtering (SVD) and Weighted Popularity algorithms to solve the cold-start problem and deliver personalized content strategies.

📋 Table of Contents
Project Overview

Key Features

System Architecture

Business Insights

Tech Stack

Installation & Setup

Project Structure

Data Source

🔭 Project Overview
The streaming industry faces two critical challenges: Information Overload for users and Content Value Decay for platforms.

MovieRec addresses these by:

Personalizing Experience: Using Latent Factor Models (SVD) to uncover hidden user taste profiles.


Solving Cold Start: Implementing a dynamic Weighted Popularity algorithm with genre filtering for new users.


Strategic Analytics: Providing a dashboard to analyze long-tail distribution and content retention value.

🚀 Key Features
1. Hybrid Recommendation Engine
New User (Cold Start): Uses a Weighted Popularity Score (IMDB formula) combined with user-selected genre filtering to recommend high-confidence trending movies.


Existing User (Personalized): Deploys a pre-computed SVD (Singular Value Decomposition) model to predict 5-star matches based on latent user-item interaction patterns.

2. Interactive Analytics Dashboard
Content Value Decay: Visualizes how "Classic" movies (1980-2000) retain higher average ratings compared to modern releases.


Long Tail Analysis: Clamped histograms showing the distribution of ratings per movie and user activity.

Quality Segmentation: Percentage breakdown of "Best" (4.5+), "Average", and "Bad" content in the catalog.

3. Explainable AI (XAI)
Contextual Explanations: For existing users, the system explains recommendations by referencing their specific viewing history (e.g., "Because you liked 'The Matrix'...").

🏗 System Architecture
The project follows a batch-processing deployment pipeline to handle resource constraints on the cloud.

Code snippet

graph LR
    A[Raw Data (MovieLens)] --> B(Preprocessing & Cleaning)
    B --> C{Model Training}
    C -->|Model 1| D[Weighted Popularity Logic]
    C -->|Model 3| E[SVD Matrix Factorization]
    E --> F[Batch Prediction Generation]
    F --> G[Parquet/CSV Export]
    D --> H[Streamlit App]
    G --> H
    H --> I[End User Interface]
Offline Training: Models are trained in Google Colab using scikit-surprise.

Batch Inference: Top-N recommendations for active users are pre-computed and stored in optimized Parquet/CSV files.

Online Inference: The Streamlit app loads lightweight artifacts to serve recommendations instantly.

📊 Business Insights
Our analysis of the MovieLens dataset revealed strategic insights for content acquisition:

The "Classic" Advantage: Movies released between 1980–2000 show significantly higher rating stability than post-2010 releases, suggesting a high ROI for acquiring library content.


The Long Tail: 90% of movies have fewer than 50 ratings, indicating a massive inventory of "Hidden Gems" that require algorithmic surfacing.

User Polarity: Users tend to be "Harsh Critics" or "Super Fans," with very few rating in the middle ground (3.0 stars).

💻 Tech Stack
Frontend: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Surprise (SVD), Scikit-Learn (Random Forest)

Visualization: Altair, Matplotlib, Seaborn

Data Storage: Parquet (for compression)

⚙️ Installation & Setup
To run this project locally:

Clone the repository:

Bash

git clone https://github.com/YourUsername/MovieRec.git
cd MovieRec
Install dependencies:

Bash

pip install -r requirements.txt
Run the App:

Bash

streamlit run app.py or python -m streamlit run app.py
📂 Project Structure
Plaintext

MovieRec/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── movies.parquet             # Optimized Movie Metadata
├── ratings.parquet            # Optimized User Ratings
├── user_recommendations.csv   # Pre-computed SVD predictions
├── README.md                  # Project Documentation
├── final_ml_team.py           # Python version of Colab notebook for Training & Analysis
└── Final_ML_Team.ipynb        # Colab notebook for Training & Analysis
💾 Data Source
This project uses the MovieLens (ml-latest) dataset provided by GroupLens Research.

Ratings: 5-star scale from 330,975 users.

Movies: 86,537 titles with genres.

Genome: Tag relevance scores for content-based filtering.

Note: Large raw CSV files (ratings.csv) are excluded from the repo. We use optimized Parquet files for deployment.

👥 Contributors (Group 8)
Irene - Lead Developer 

Joshua Agyemang - Data Analyst

Velma Atieno - Research & Documentation

Chika Amanna @AmannaP - ML Engineer

📜 License
This project is for educational purposes under the MIT License. Dataset usage is governed by the Group8 Usage License.
