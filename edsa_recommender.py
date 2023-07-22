"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
from PIL import Image
import joblib,os


# Data handling dependencies
import pandas as pd
import numpy as np
import utils
import base64

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

#Loading svd model

@st.cache_resource
def load_model(url, name):
	SVD= open(url, name) 
	model = joblib.load(SVD) # load SVD from the pkl file
	return model


SVD = load_model("resources\models\SVD.pkl","rb")

# App declaration
def main():
    
    
    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","About Us", "EDA","How It Works","FAQs","Feedback"]
    st.sidebar.image('resources\imgs\log.jpg',use_column_width=True)
               
    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        file_ = open('resources\imgs\GIF2.gif', "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="GIF gif">',
        unsafe_allow_html=True,)
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image('resources\imgs\gg.jpg',use_column_width=True)
        st.write("Describe your winning approach on this page")
    if  page_selection == "About Us":
        st.title("About Us")
        st.write("[Meet the team>]")
        st.write("What we do")
    if  page_selection=="EDA":
        st.title("EDA")
        st.write("The following EDA was done prior to developing the reccomemder algorithm")
    if  page_selection=="How It Works":
        st.title("How It Works")
        st.info("Simple Explanation")
        expander = st.expander("See here for more info")
        expander.write("This is a non-technical explanation explanation on how the app works.")
        st.image('resources\imgs\content-based_vs_collaborative_light.png',use_column_width=True)
        st.info("Complicated Explanation")
        expand = st.expander("See here for more info")
        expand.write('''This is a technical explanation on how the app works''')

    if  page_selection == "FAQs":    
        st.info("Frequently Asked Questions (FAQs) for the movie recommender App")
        faq_one = st.expander("What is the purpose of the ML Tech movie recommender App?")
        faq_one.write('''The purpose of this app is to recommend movies to viewers based on the movies they have watched before.''')
        faq_two = st.expander("How does the app recommend movies?")
        faq_two.write('''The app uses a machine learning model .................''')
        faq_three=st.expander("What technologies are used to build the app?") 
        faq_three.write('''The app is built using the Streamlit framework, which is a Python library for building interactive web applications.
It also leverages machine learning libraries such as scikit-learn for the recommender model.''')
        faq_four=st.expander("How accurate is the movie recommender?")
        faq_four.write('''The accuracy of the recommender model depends on........''')
        faq_five=st.expander("Can the app handle real-time data?") 
        faq_five.write('''Yes, the app can handle real-time data...............''')
        faq_six=st.expander("Is the app user-friendly for non-technical users?")
        faq_six.write('''Yes, the app is designed to be user-friendly even for non-technical users. 
The Streamlit framework simplifies the development process and provides an intuitive interface where users can easily pick their favourite movies and have the app recommend movies they are more likely to watch in just one press.''') 
        faq_sev=st.expander("Are there any limitations to the app?")
        faq_sev.write('''The limitations of the app depend on the performance of the underlying machine learning model. Some potential limitations could include:...............''')
        faq_eight= st.expander(" Can the app be customized or extended?")
        faq_eight.write('''Yes, the app can be customized or extended based on specific requirements.
Additional features, visualizations, or modifications can be implemented using the flexibility provided by the Streamlit framework.''')
        faq_nine = st.expander("What are the system requirements to run the app?")
        faq_nine.write('''The app requires a compatible Python environment with the necessary dependencies installed. Users will need to have access to the internet to retrieve the required data and resources.''')

    if  page_selection=="Feedback":
        st.title("Feedback")
        # Feedback form
# Likert scale
        slider =st.select_slider("How satisfied are you with this app?", options=["Very Dissatisfied", "Somewhat Dissatisfied", "Neutral", "Somewhat Satisfied", "Very Satisfied"])
        st.write("We would love to hear your feedback and suggestions")
        feedback_button = st.button("Submit Feedback")
        #Save the feedback to a file, database, or process it as needed
        st.success("Thank you for your feedback!")     

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    
      
if __name__ == '__main__':
    main()
