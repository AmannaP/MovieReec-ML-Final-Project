import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MovieRec Analytics",
    page_icon="🎬",
    layout="wide"
)

# --- CUSTOM CSS FOR COLOR THEME & FIXES ---
st.markdown("""
    <style>
    /* 1. Main Page Background - Midnight Gradient */
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* 2. Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0f2027;
    }
    
    /* 3. General Text Color (Headers, Paragraphs) */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* 4. FIX: Dropdowns, Multiselects, and Text Inputs */
    .stSelectbox div, .stMultiSelect div, .stTextInput div {
        color: #000000 !important;
    }
    
    /* Specific fix for the text input cursor and typed text */
    input[type="text"], input[type="number"] {
        color: #000000 !important; 
        background-color: #ffffff !important;
    }
    
    /* Fix for the dropdown popup menu items */
    ul[data-testid="stVirtualDropdown"] li {
        color: #000000 !important;
    }
    
    /* 5. Metrics and Dataframes */
    [data-testid="stMetricValue"] {
        color: #ffffff;
    }
    
    /* Make Dataframes readable */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.9); 
        color: #000000 !important;
    }
    
    /* 6. Hyperlinks */
    a {
        color: #4da6ff !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # NEW PARQUET WAY
        recs_df = pd.read_parquet('user_recommendations.parquet')
        movies_df = pd.read_parquet('movies.parquet')
        ratings_df = pd.read_parquet('ratings.parquet')

        # --- PRE-PROCESSING: EXTRACT YEAR GLOBALLY ---
        def extract_year(t): 
            match = re.search(r'\((\d{4})\)', str(t))
            return int(match.group(1)) if match else 2000 # Default to 2000 if no year found
        
        movies_df['year'] = movies_df['title'].apply(extract_year)

        return recs_df, movies_df, ratings_df
    except FileNotFoundError:
        # Fallback to CSV
        try:
            recs_df = pd.read_csv('user_recommendations.csv')
            movies_df = pd.read_csv('movies.csv')
            ratings_df = pd.read_csv('ratings.csv')

            # --- PRE-PROCESSING: EXTRACT YEAR GLOBALLY ---
            def extract_year(t): 
                match = re.search(r'\((\d{4})\)', str(t))
                return int(match.group(1)) if match else 2000 # Default to 2000 if no year found
            
            movies_df['year'] = movies_df['title'].apply(extract_year)
            
            return recs_df, movies_df, ratings_df
        except:
            st.error("Data files not found. Please ensure .parquet or .csv files are uploaded.")
            return None, None, None

recs_df, movies, ratings = load_data()

# --- HELPER FUNCTIONS ---

def get_weighted_popularity(df, min_vote_percentile=0.90):
    # Model A: Weighted Popularity
    C = df['rating'].mean()
    stats = df.groupby('movieId').agg({'rating': ['count', 'mean']})
    stats.columns = ['vote_count', 'vote_average']
    
    m = stats['vote_count'].quantile(min_vote_percentile)
    qualified = stats[stats['vote_count'] >= m].copy()
    
    qualified['weighted_score'] = (qualified['vote_count'] / (qualified['vote_count'] + m) * qualified['vote_average']) + \
                                  (m / (qualified['vote_count'] + m) * C)
    
    return qualified.sort_values('weighted_score', ascending=False)

def get_content_based_recs(selected_genres, movies_df, n=10):
    # Model B: Content-Based
    mask = pd.Series([True] * len(movies_df))
    for genre in selected_genres:
        mask = mask & movies_df['genres'].str.contains(genre, regex=False)
        
    filtered_movies = movies_df[mask]
    
    if len(filtered_movies) < n:
        pattern = '|'.join(selected_genres)
        filtered_movies = movies_df[movies_df['genres'].str.contains(pattern, regex=True)]
    
    if not filtered_movies.empty:
        return filtered_movies.sample(min(n, len(filtered_movies)))
    else:
        return pd.DataFrame()

def get_user_history_text(user_id, ratings_df, movies_df):
    user_history = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_history[user_history['rating'] >= 4.0]
    
    if liked_movies.empty:
        liked_movies = user_history.sort_values('rating', ascending=False).head(3)
    
    liked_movies_with_titles = liked_movies.merge(movies_df, on='movieId')
    
    if not liked_movies_with_titles.empty:
        titles = liked_movies_with_titles['title'].sample(min(3, len(liked_movies_with_titles))).values
        titles_str = ", ".join([f"**{t}**" for t in titles])
        return f"User {user_id} liked movies like {titles_str} and gave them high ratings."
    else:
        return f"User {user_id} has a unique taste profile."

# --- APP UI LAYOUT ---

# Sidebar Customization
st.sidebar.markdown("# 📽️ MovieRec Platform")
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analytics Dashboard", "Recommender System"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("🎬 MovieRec Intelligent Recommender")
    st.markdown("### Group 8 Machine Learning Final Project")
    
    st.markdown("""
    This application demonstrates a **Hybrid Recommendation Engine** designed to solve the Cold Start problem while providing high-accuracy personalized predictions.
    """)
    
    # SYSTEM ARCHITECTURE DIAGRAM
    st.markdown("#### 🏗️ System Architecture")
    st.markdown("""
    ```mermaid
    graph LR
        A[User Arrives] --> B{New User?}
        B -- Yes --> C[Cold Start Strategy]
        B -- No --> D[Warm Start Strategy]
        C --> E(Model A: Popularity: Using IMDB Popularity Formular)
        C --> F(Model B: Content Filter: Using RandomForest Classifier)
        D --> G(Model C: SVD Matrix Factorization: Personalized Recommendations)
        E --> H[Recommendations]
        F --> H
        G --> H
    ```
        IMDB Popularity Formular: (v / (v+m) * R) + (m / (v+m) * C))
                where v = number of votes for the movie
                      m = minimum votes required to be listed in the chart
                      R = average rating of the movie
                      C = mean vote across the whole report
                
    """)
    st.info("👈 Select **'Recommender System'** in the sidebar to test the models.")

# --- PAGE 2: ANALYTICS DASHBOARD ---
elif page == "Analytics Dashboard":
    st.title("📊 Business Insights")
    
    if ratings is not None:
        # Define 5 Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Genre Trends", 
            "Content Value Decay", 
            "Movie Popularity", 
            "User Activity", 
            "Rating Quality"
        ])

        # 1. GENRE TRENDS GRAPH
        with tab1:
            st.subheader("The Rise and Fall of Movie Genres")
            st.markdown("Visualizing which genres dominated the box office over the decades.")
            
            movies_exploded = movies.assign(genres=movies['genres'].str.split('|')).explode('genres')
            genre_trends = movies_exploded[movies_exploded['year'] >= 1980].merge(ratings, on='movieId')
            genre_yearly = genre_trends.groupby(['year', 'genres']).size().reset_index(name='count')
            
            top_genres = genre_yearly.groupby('genres')['count'].sum().nlargest(7).index
            genre_yearly_filtered = genre_yearly[genre_yearly['genres'].isin(top_genres)]
            
            chart_trends = alt.Chart(genre_yearly_filtered).mark_area(opacity=0.6).encode(
                x=alt.X('year:O', title='Year'),
                y=alt.Y('count:Q', stack='center', title='Number of Ratings'),
                color=alt.Color('genres:N', title='Genre'),
                tooltip=['year', 'genres', 'count']
            ).properties(height=400)
            
            st.altair_chart(chart_trends, use_container_width=True)

        # 2. Content Value Decay
        with tab2:
            st.subheader("Do older movies hold their value?")
            merged = ratings.merge(movies, on='movieId')
            year_stats = merged[(merged['year'] >= 1980) & (merged['year'] <= 2020)].groupby('year')['rating'].mean()
            st.line_chart(year_stats)
            st.success("INSIGHT: 'Classic' movies (1980-2000) show higher average ratings than recent releases.")

        # 3. Movie Popularity
        with tab3:
            st.subheader("Long Tail Distribution")
            movie_counts = ratings.groupby('movieId')['rating'].count().reset_index()
            movie_counts.columns = ['movieId', 'rating_count']
            
            chart = alt.Chart(movie_counts).mark_bar().encode(
                x=alt.X('rating_count', bin=alt.Bin(maxbins=30, extent=[0, 50]), title='Number of Ratings Received (Zoomed to 0-50)'),
                y=alt.Y('count()', title='Number of Movies')
            ).properties(title="Most movies have fewer than 20 ratings")
            st.altair_chart(chart, use_container_width=True)

        # 4. User Activity
        with tab4:
            st.subheader("Distribution of Ratings per User")
            user_counts = ratings.groupby('userId')['rating'].count().reset_index()
            user_counts.columns = ['userId', 'rating_count']
            
            chart_users = alt.Chart(user_counts).mark_bar(color='orange').encode(
                x=alt.X('rating_count', bin=alt.Bin(maxbins=30, extent=[0, 100]), title='Ratings Given (Zoomed to 0-100)'),
                y=alt.Y('count()', title='Number of Users')
            ).properties(title="User Activity: Most users rate fewer than 100 movies")
            st.altair_chart(chart_users, use_container_width=True)

        # 5. Rating Quality
        with tab5:
            st.subheader("Movie Quality Segmentation")
            avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
            
            def categorize_rating(r):
                if r >= 3.5: return "Generous (3.5 - 5.0)"
                elif r >= 2.5: return "Average (2.5 - 3.5)"
                else: return "Harsh (0 - 2.5)"
                
            avg_ratings['Category'] = avg_ratings['rating'].apply(categorize_rating)
            category_counts = avg_ratings['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            total_movies = category_counts['Count'].sum()
            category_counts['Percentage'] = (category_counts['Count'] / total_movies * 100).round(1)
            category_counts['Label'] = category_counts['Category'] + " (" + category_counts['Percentage'].astype(str) + "%)"
            
            chart_qual = alt.Chart(category_counts).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Label", type="nominal", title="Rating Category", 
                                scale=alt.Scale(domain=[
                                    f"Generous (3.5 - 5.0) ({category_counts.loc[category_counts['Category']=='Generous (3.5 - 5.0)', 'Percentage'].values[0]}%)",
                                    f"Average (2.5 - 3.5) ({category_counts.loc[category_counts['Category']=='Average (2.5 - 3.5)', 'Percentage'].values[0]}%)",
                                    f"Harsh (0 - 2.5) ({category_counts.loc[category_counts['Category']=='Harsh (0 - 2.5)', 'Percentage'].values[0]}%)"
                                ], range=['#4CAF50', '#FFC107', '#F44336'])),
                tooltip=["Category", "Count", "Percentage"]
            ).properties(title="Proportion of Movie Quality")
            
            st.altair_chart(chart_qual, use_container_width=True)

# --- PAGE 3: RECOMMENDER SYSTEM ---
elif page == "Recommender System":
    st.title("🍿 Movie Recommender Engine")
    
    user_segment = st.radio("Select User Segment:", ["New User (Cold Start)", "Existing User (Personalized)"], horizontal=True)
    
    # --- SCENARIO A: NEW USER ---
    if user_segment == "New User (Cold Start)":
        st.subheader("❄️ Cold Start Strategy")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_choice = st.selectbox(
                "Select Prediction Model:",
                ["Model A: Weighted Popularity (Trending)", "Model B: Content-Based (Genre Matcher)"]
            )
        
        # YEAR FILTER
        min_year = int(movies['year'].min())
        max_year = int(movies['year'].max())
        
        with col2:
            year_range = st.slider(
                "Filter by Release Year:", 
                min_value=1950, 
                max_value=max_year, 
                value=(1980, 2020),
                key="new_user_year_slider" # Unique Key
            )

        unique_genres = set()
        movies['genres'].str.split('|').apply(unique_genres.update)
        sorted_genres = sorted(list(unique_genres))
        
        selected_genres = st.multiselect("Select genres (Optional for Model A, Required for Model B):", sorted_genres, max_selections=3)
        
        if st.button("Generate Recommendations"):
            
            # Apply Year Filter
            movies_filtered = movies[(movies['year'] >= year_range[0]) & (movies['year'] <= year_range[1])]
            
            if movies_filtered.empty:
                st.error("No movies found in this year range! Try widening the years.")
            else:
                # --- MODEL A ---
                if model_choice == "Model A: Weighted Popularity (Trending)":
                    merged_df = ratings.merge(movies_filtered, on='movieId')
                    ranked_movies = get_weighted_popularity(merged_df)
                    ranked_with_info = ranked_movies.merge(movies, on='movieId')
                    
                    if selected_genres:
                        pattern = '|'.join(selected_genres)
                        final_recs = ranked_with_info[ranked_with_info['genres'].str.contains(pattern, regex=True)]
                    else:
                        final_recs = ranked_with_info

                    st.success("Using **Weighted Popularity Algorithm**.")
                    
                    for i, (index, row) in enumerate(final_recs.head(10).iterrows()):
                        movie_title = row['title']
                        google_link = f"https://www.google.com/search?q={movie_title.replace(' ', '+')}+movie"
                        
                        st.markdown(f"**{i+1}. [{movie_title}]({google_link})**")
                        st.caption(f"📅 {int(row['year'])} | {row['genres']} | ⭐ {row['vote_average']:.1f}")
                        st.divider()

                # --- MODEL B ---
                elif model_choice == "Model B: Content-Based (Genre Matcher)":
                    if not selected_genres:
                        st.error("⚠️ For Content-Based matching, you MUST select at least one genre.")
                    else:
                        final_recs = get_content_based_recs(selected_genres, movies_filtered, n=10)
                        
                        if final_recs.empty:
                            st.warning("No movies found matching those genres in that year range.")
                        else:
                            st.success("Using **Content-Based Filtering**.")
                            for i, (index, row) in enumerate(final_recs.head(10).iterrows()):
                                movie_title = row['title']
                                google_link = f"https://www.google.com/search?q={movie_title.replace(' ', '+')}+movie"
                                
                                st.markdown(f"**{i+1}. [{movie_title}]({google_link})**")
                                st.caption(f"📅 {int(row['year'])} | {row['genres']}")
                                st.divider()

    # --- SCENARIO B: EXISTING USER (UPDATED) ---
    elif user_segment == "Existing User (Personalized)":
        st.subheader("🔥 Personalized Strategy (SVD Model)")
        
        if recs_df is not None:
            
            # Layout Columns
            col_id, col_year = st.columns([1, 1])
            
            with col_id:
                selected_user = st.number_input(
                    "Enter User ID (Type a number from 0 to 4000):", 
                    min_value=1, 
                    max_value=10000, 
                    value=1,
                    step=1
                )
            
            # --- NEW: YEAR FILTER FOR EXISTING USERS ---
            min_year = int(movies['year'].min())
            max_year = int(movies['year'].max())
            
            with col_year:
                year_range_existing = st.slider(
                    "Filter by Release Year:", 
                    min_value=1950, 
                    max_value=max_year, 
                    value=(1980, 2020),
                    key="existing_user_year_slider" # Unique Key
                )
            
            if st.button(f"Analyze User {selected_user}"):
                if selected_user in recs_df['userId'].values:
                    history_text = get_user_history_text(selected_user, ratings, movies)
                    st.info(f"**Analysis:** {history_text}")
                    st.markdown("Therefore, we recommend the following movies based on your taste profile:")
                    
                    user_recs = recs_df[recs_df['userId'] == selected_user]
                    
                    # --- CRITICAL STEP: MERGE YEAR INFO FOR FILTERING ---
                    # The parquet file usually doesn't have 'year', so we add it now
                    user_recs = user_recs.merge(movies[['movieId', 'year']], on='movieId', how='left')
                    
                    # --- APPLY FILTER ---
                    user_recs_filtered = user_recs[
                        (user_recs['year'] >= year_range_existing[0]) & 
                        (user_recs['year'] <= year_range_existing[1])
                    ]
                    
                    if user_recs_filtered.empty:
                        st.warning(f"User {selected_user} has recommendations, but none are from {year_range_existing[0]}-{year_range_existing[1]}. Try widening the year range.")
                    else:
                        for i, (index, row) in enumerate(user_recs_filtered.iterrows()):
                            movie_title = row['title']
                            google_link = f"https://www.google.com/search?q={movie_title.replace(' ', '+')}+movie"
                            
                            st.markdown(f"**{row['rank']}. [{movie_title}]({google_link})**")
                            # Now we can safely display the year
                            st.caption(f"📅 {int(row['year'])} | {row['genres']} | Predicted Rating: {row['predicted_rating']:.2f} ⭐")
                            st.divider()
                else:
                    st.error(f"⚠️ User ID {selected_user} not found in the pre-computed database.")
                    st.warning("For this demo, please ensure you generated recommendations for this User ID range in the notebook.")