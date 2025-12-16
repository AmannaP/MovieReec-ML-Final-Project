import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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
    
    /* 3. General Text Color (Headers, Paragraphs) - EXCLUDING List Items (li) to avoid breaking dropdowns */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* 4. FIX: Dropdowns, Multiselects, and Text Inputs */
    /* Forces the text INSIDE the input box and the dropdown menu to be BLACK */
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
        return recs_df, movies_df, ratings_df
    except FileNotFoundError:
        # Fallback to CSV
        try:
            recs_df = pd.read_csv('user_recommendations.csv')
            movies_df = pd.read_csv('movies.csv')
            ratings_df = pd.read_csv('ratings.csv')
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
    liked_movies = user_history[user_history['rating'] >= 4.5]
    
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
    st.markdown("""
    ### Project Overview: MovieRec
    This application demonstrates a Hybrid Recommendation Engine for Group 8 Machine Learning class team final project.
    
    **Architecture:**
    * **Cold Start (New Users):** Choice between **Weighted Popularity** (Trending) and **Content-Based** (Genre Matching).
    * **Warm Start (Existing Users):** **SVD Matrix Factorization** (Personalized).
    """)
    st.info("👈 Select 'Recommender System' to test the models.")

# --- PAGE 2: ANALYTICS DASHBOARD ---
elif page == "Analytics Dashboard":
    st.title("📊 Business Insights")
    
    if ratings is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["Content Value Decay", "Movie Popularity", "User Activity", "Rating Quality"])

        # 1. Content Value Decay
        with tab1:
            st.subheader("Do older movies hold their value?")
            import re
            def extract_year(t): 
                match = re.search(r'\((\d{4})\)', str(t))
                return int(match.group(1)) if match else None
            
            movies['year'] = movies['title'].apply(extract_year)
            merged = ratings.merge(movies, on='movieId')
            year_stats = merged[(merged['year'] >= 1980) & (merged['year'] <= 2020)].groupby('year')['rating'].mean()
            st.line_chart(year_stats)
            st.success("INSIGHT: 'Classic' movies (1980-2000) show higher average ratings than recent releases.")

        # 2. Movie Popularity
        with tab2:
            st.subheader("Distribution of Ratings per Movie")
            movie_counts = ratings.groupby('movieId')['rating'].count().reset_index()
            movie_counts.columns = ['movieId', 'rating_count']
            
            chart = alt.Chart(movie_counts).mark_bar().encode(
                x=alt.X('rating_count', bin=alt.Bin(maxbins=30, extent=[0, 50]), title='Number of Ratings Received (Zoomed to 0-50)'),
                y=alt.Y('count()', title='Number of Movies')
            ).properties(title="Long Tail Distribution: Most movies have fewer than 20 ratings")
            st.altair_chart(chart, use_container_width=True)

        # 3. User Activity
        with tab3:
            st.subheader("Distribution of Ratings per User")
            user_counts = ratings.groupby('userId')['rating'].count().reset_index()
            user_counts.columns = ['userId', 'rating_count']
            
            chart_users = alt.Chart(user_counts).mark_bar(color='orange').encode(
                x=alt.X('rating_count', bin=alt.Bin(maxbins=30, extent=[0, 100]), title='Ratings Given (Zoomed to 0-100)'),
                y=alt.Y('count()', title='Number of Users')
            ).properties(title="User Activity: Most users rate fewer than 100 movies")
            st.altair_chart(chart_users, use_container_width=True)

        # 4. Rating Quality
        with tab4:
            st.subheader("Movie Quality Segmentation")
            avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
            
            def categorize_rating(r):
                if r >= 3.5: return "Best (3.5 - 5.0)"
                elif r >= 2.5: return "Average (2.5 - 3.5)"
                else: return "Bad (0 - 2.5)"
                
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
                                    f"Best (3.5 - 5.0) ({category_counts.loc[category_counts['Category']=='Best (3.5 - 5.0)', 'Percentage'].values[0]}%)",
                                    f"Average (2.5 - 3.5) ({category_counts.loc[category_counts['Category']=='Average (2.5 - 3.5)', 'Percentage'].values[0]}%)",
                                    f"Bad (0 - 2.5) ({category_counts.loc[category_counts['Category']=='Bad (0 - 2.5)', 'Percentage'].values[0]}%)"
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
        
        model_choice = st.selectbox(
            "Select Prediction Model:",
            ["Model A: Weighted Popularity (Trending)", "Model B: Content-Based (Genre Matcher)"]
        )
        
        unique_genres = set()
        movies['genres'].str.split('|').apply(unique_genres.update)
        sorted_genres = sorted(list(unique_genres))
        
        selected_genres = st.multiselect("Select genres (Optional for Model A, Required for Model B):", sorted_genres, max_selections=3)
        
        if st.button("Generate Recommendations"):
            if model_choice == "Model A: Weighted Popularity (Trending)":
                merged_df = ratings.merge(movies, on='movieId')
                ranked_movies = get_weighted_popularity(merged_df)
                ranked_with_info = ranked_movies.merge(movies, on='movieId')
                
                if selected_genres:
                    pattern = '|'.join(selected_genres)
                    final_recs = ranked_with_info[ranked_with_info['genres'].str.contains(pattern, regex=True)]
                else:
                    final_recs = ranked_with_info

                st.success("Using **Weighted Popularity Algorithm** (Statistical Model).")
                for i, (index, row) in enumerate(final_recs.head(10).iterrows()):
                    st.write(f"**{i+1}. {row['title']}**")
                    st.caption(f"Genre: {row['genres']} | ⭐ Avg: {row['vote_average']:.1f}")
                    st.divider()

            elif model_choice == "Model B: Content-Based (Genre Matcher)":
                if not selected_genres:
                    st.error("⚠️ For Content-Based matching, you MUST select at least one genre.")
                else:
                    final_recs = get_content_based_recs(selected_genres, movies, n=10)
                    st.success("Using **Content-Based Filtering** (Random Forest Logic).")
                    for i, (index, row) in enumerate(final_recs.head(10).iterrows()):
                        st.write(f"**{i+1}. {row['title']}**")
                        st.caption(f"Genre: {row['genres']}")
                        st.divider()

    # --- SCENARIO B: EXISTING USER ---
    elif user_segment == "Existing User (Personalized)":
        st.subheader("🔥 Personalized Strategy (SVD Model)")
        
        if recs_df is not None:
            selected_user = st.number_input(
                "Enter User ID (Type a number from 0 to 4000):", 
                min_value=1, 
                max_value=10000, 
                value=1,
                step=1
            )
            
            if st.button(f"Analyze User {selected_user}"):
                if selected_user in recs_df['userId'].values:
                    history_text = get_user_history_text(selected_user, ratings, movies)
                    st.info(f"**Analysis:** {history_text}")
                    st.markdown("Therefore, we recommend the following movies based on your taste profile:")
                    
                    user_recs = recs_df[recs_df['userId'] == selected_user]
                    st.table(user_recs[['rank', 'title', 'genres', 'predicted_rating']].set_index('rank'))
                else:
                    st.error(f"⚠️ User ID {selected_user} not found in the pre-computed database.")
                    st.warning("For this demo, please ensure you generated recommendations for this User ID range in the notebook.")