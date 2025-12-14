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

# --- LOAD DATA ---
# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # NEW PARQUET WAY (Requires 'pip install pyarrow' or 'fastparquet')
        # Load the pre-computed recommendations
        recs_df = pd.read_parquet('user_recommendations.parquet')
        # Load raw movies for the "New User" logic
        movies_df = pd.read_parquet('movies.parquet')
        ratings_df = pd.read_parquet('ratings.parquet')
        return recs_df, movies_df, ratings_df
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please ensure you uploaded the .parquet files.")
        return None, None, None

recs_df, movies, ratings = load_data()

# --- HELPER FUNCTIONS ---

def get_weighted_popularity(df, min_vote_percentile=0.90):
    # Model 1: Weighted Popularity (Calculated live)
    C = df['rating'].mean()
    stats = df.groupby('movieId').agg({'rating': ['count', 'mean']})
    stats.columns = ['vote_count', 'vote_average']
    
    # Filter for movies with enough votes
    m = stats['vote_count'].quantile(min_vote_percentile)
    qualified = stats[stats['vote_count'] >= m].copy()
    
    # Calculate Score
    qualified['weighted_score'] = (qualified['vote_count'] / (qualified['vote_count'] + m) * qualified['vote_average']) + \
                                  (m / (qualified['vote_count'] + m) * C)
    
    return qualified.sort_values('weighted_score', ascending=False)

def get_user_history_text(user_id, ratings_df, movies_df):
    # Finds highly rated movies by this user to generate explanatory text
    user_history = ratings_df[ratings_df['userId'] == user_id]
    
    # Get movies rated 4.5 or higher
    liked_movies = user_history[user_history['rating'] >= 4.5]
    
    # Fallback to just "highest rated" if they are a harsh critic
    if liked_movies.empty:
        liked_movies = user_history.sort_values('rating', ascending=False).head(3)
    
    # Merge to get titles
    liked_movies_with_titles = liked_movies.merge(movies_df, on='movieId')
    
    # Pick up to 3 random titles to show
    if not liked_movies_with_titles.empty:
        titles = liked_movies_with_titles['title'].sample(min(3, len(liked_movies_with_titles))).values
        titles_str = ", ".join([f"**{t}**" for t in titles])
        return f"User {user_id} liked movies like {titles_str} and gave them high ratings."
    else:
        return f"User {user_id} has a unique taste profile."

# --- APP UI LAYOUT ---

# Sidebar Customization (REMOVED NETFLIX LOGO)
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
    * **Model 1 (Cold Start):** Weighted Popularity with Genre Filtering.
    * **Model 3 (Personalized):** SVD Matrix Factorization (Batch Deployment).
    """)
    st.info("👈 Select 'Recommender System' to test the models.")

# --- PAGE 2: ANALYTICS DASHBOARD ---
elif page == "Analytics Dashboard":
    st.title("📊 Business Insights")
    
    # Prepare Data for Charts
    if ratings is not None:
        
        # --- TAB STRUCTURE ---
        tab1, tab2, tab3, tab4 = st.tabs(["Content Value Decay", "Movie Popularity", "User Activity", "Rating Quality"])

        # 1. Content Value Decay (Existing)
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

        # 2. Graph of movies with their number of ratings (Fixed Range)
        with tab2:
            st.subheader("Distribution of Ratings per Movie")
            movie_counts = ratings.groupby('movieId')['rating'].count().reset_index()
            movie_counts.columns = ['movieId', 'rating_count']
            
            # IMPROVEMENT: Clamp the X-axis to 0-50 to avoid outliers stretching the graph
            # Movies with >50 ratings will still be counted but grouped visually
            chart = alt.Chart(movie_counts).mark_bar().encode(
                x=alt.X('rating_count', bin=alt.Bin(maxbins=30, extent=[0, 50]), title='Number of Ratings Received (Zoomed to 0-50)'),
                y=alt.Y('count()', title='Number of Movies')
            ).properties(
                title="Long Tail Distribution: Most movies have fewer than 20 ratings"
            )
            st.altair_chart(chart, use_container_width=True)

        # 3. Users with the ratings (Fixed Range)
        with tab3:
            st.subheader("Distribution of Ratings per User")
            user_counts = ratings.groupby('userId')['rating'].count().reset_index()
            user_counts.columns = ['userId', 'rating_count']
            
            # IMPROVEMENT: Clamp the X-axis to 0-100 to show normal user behavior
            chart_users = alt.Chart(user_counts).mark_bar(color='orange').encode(
                x=alt.X('rating_count', bin=alt.Bin(maxbins=30, extent=[0, 100]), title='Ratings Given (Zoomed to 0-100)'),
                y=alt.Y('count()', title='Number of Users')
            ).properties(
                title="User Activity: Most users rate fewer than 100 movies"
            )
            st.altair_chart(chart_users, use_container_width=True)

        # 4. Movies by Rating Category (Added Percentages)
        with tab4:
            st.subheader("Movie Quality Segmentation")
            # Calculate average rating per movie
            avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
            
            # Segment function
            def categorize_rating(r):
                if r >= 3.5: return "Best (3.5 - 5.0)"
                elif r >= 2.5: return "Average (2.5 - 3.5)"
                else: return "Bad (0 - 2.5)"
                
            avg_ratings['Category'] = avg_ratings['rating'].apply(categorize_rating)
            category_counts = avg_ratings['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Calculate Percentage
            total_movies = category_counts['Count'].sum()
            category_counts['Percentage'] = (category_counts['Count'] / total_movies * 100).round(1)
            
            # Create Label column for the Legend (e.g., "Bad (35%)")
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
    
    user_type = st.radio("Select User Type:", ["New User (Cold Start)", "Existing User (Personalized)"])
    
    # --- SCENARIO A: NEW USER ---
    if user_type == "New User (Cold Start)":
        st.subheader("🔥 Top Trending by Genre")
        
        # 1. Extract Genres for Dropdown
        # Flatten the pipe-separated genres into a unique list
        unique_genres = set()
        movies['genres'].str.split('|').apply(unique_genres.update)
        sorted_genres = sorted(list(unique_genres))
        
        # 2. User Input
        selected_genres = st.multiselect("Step 1: Select up to 2 genres you like:", sorted_genres, max_selections=2)
        
        if st.button("Generate Recommendations"):
            if not selected_genres:
                st.warning("Please select at least one genre!")
            else:
                # Calculate Weighted Popularity Globally first
                merged_df = ratings.merge(movies, on='movieId')
                ranked_movies = get_weighted_popularity(merged_df)
                
                # Merge titles and genres back
                ranked_with_info = ranked_movies.merge(movies, on='movieId')
                
                # Filter by Selected Genres
                # Check if the movie's genre string contains ANY of the selected genres
                pattern = '|'.join(selected_genres)
                filtered_recs = ranked_with_info[ranked_with_info['genres'].str.contains(pattern, regex=True)]
                
                st.success(f"Showing Top Trending Movies for: {', '.join(selected_genres)}")
                
                # Display Top 10
                for i, (index, row) in enumerate(filtered_recs.head(10).iterrows()):
                    st.write(f"**{i+1}. {row['title']}**")
                    st.caption(f"Genre: {row['genres']} | ⭐ Avg Rating: {row['vote_average']:.1f} ({int(row['vote_count'])} votes)")
                    st.divider()

    # --- SCENARIO B: EXISTING USER ---
    elif user_type == "Existing User (Personalized)":
        st.subheader("👤 Personalized For You (SVD Model)")
        
        # Get list of users we have pre-computed recommendations for
        if recs_df is not None:
            available_users = recs_df['userId'].unique()
            selected_user = st.selectbox("Select User Profile ID:", available_users)
            
            if st.button(f"Get Picks for User {selected_user}"):
                
                # 1. Generate Explanation Text (The "Why")
                history_text = get_user_history_text(selected_user, ratings, movies)
                st.markdown(f"**Analysis:** {history_text}")
                st.markdown("Therefore, we recommend the following movies based on your taste profile:")
                
                # 2. Filter the CSV for this user
                user_recs = recs_df[recs_df['userId'] == selected_user]
                
                # 3. Show neat table
                st.table(user_recs[['rank', 'title', 'genres', 'predicted_rating']].set_index('rank'))