import pandas as pd
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration and load data
st.set_page_config(
    page_title="Book Recommendation App",
    page_icon="📚",
    layout="wide",
    menu_items={
        'Get Help': 'http://www.quickmeme.com/img/54/547621773e22705fcfa0e73bc86c76a05d4c0b33040fcb048375dfe9167d8ffc.jpg',
        'Report a bug': "https://w7.pngwing.com/pngs/839/902/png-transparent-ladybird-ladybird-bug-miscellaneous-presentation-insects-thumbnail.png",
        'About': "This is a Book Recommendation App. Very Easy to use!"
    }
)

# Load data
books=pd.read_csv("Books.csv",encoding='latin-1')
final_ratings = pd.read_csv("final_ratings.csv",encoding='latin-1')

# Pivot table for user-based recommendation
pt = final_ratings.pivot_table(index="bookTitle", columns="userId", values="bookRating")
pt.fillna(0, inplace=True)

# Function to recommend similar books
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similarity_scores = cosine_similarity(pt)
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    recommended_books = [pt.index[i[0]] for i in similar_items]
    return recommended_books

# Streamlit app
def main():
    st.title("Book Recommendation System")
    st.header("User-Based Recommender")

    # User input for User ID or Book Title
    user_input = st.text_input("Enter User ID or Book Title:")

    if st.button("Recommend"):
        try:
            user_input = int(user_input)
            if user_input in pt.columns:
                user_ratings = pt[user_input].dropna()
                top_rated_books = user_ratings.sort_values(ascending=False).head(5)

                # Display top-rated books for the user
                st.subheader(f"Top 5 rated books for User {user_input}:")
                st.write(top_rated_books)
            else:
                st.warning("Invalid User ID. Please enter a valid User ID.")
        except ValueError:
            if user_input in pt.index:
                # Display recommended books for the entered book title
                recommended_books = recommend(user_input)
                st.subheader(f"Books similar to '{user_input}':")
                st.write(recommended_books)
            else:
                st.warning(f"Book '{user_input}' not found in the dataset.")

    # User input for Book Title
    st.sidebar.title("Content-Based Recommender")
    book_input = st.text_input("Enter Book Name:")

    if st.button("Recommend similar books"):
        if book_input in pt.index:
            # Display recommended books for the entered book title
            recommended_books = recommend(book_input)
            st.subheader(f"Books similar to '{book_input}':")
            st.write(recommended_books)
        else:
            st.warning(f"Book '{book_input}' not found in the dataset.")

if __name__ == "__main__":
    main()
