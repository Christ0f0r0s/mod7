import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore



st.set_page_config(
    page_title="Szukaj znajomych",
    page_icon=":material/group_search:",
    layout="wide",  # lub 'centered'
    initial_sidebar_state="expanded",  # lub 'collapsed', 'auto'
    
)

st.markdown(
    """
    <style>
    /* Tło górnej belki */
    header, .css-1v3fvcr {
        background-color: #fff8ef !important;
    }
    /* Styl przycisków w st.pills */
    button[aria-selected="true"] {
        background-color: #ff4b00 !important;
        color: white !important;
    }
    /* Tło wykresów (np. Plotly) */
    .plot-container {
        background-color: #fff8ef !important;
    }
	
	 .stApp {
        background-color: #fff8ef;
    }

    /* Tło zaznaczonego przycisku w st.radio */
    div.row-widget.stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
        background-color: #8b6f47 !important;
        border-color: #8b6f47 !important;
    }
    /* Kolor tekstu zaznaczonego przycisku */
    div.row-widget.stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] span {
        color: white !important;
    }
    /* Kolor tekstu przycisków st.radio */
    div.row-widget.stRadio div[role="radiogroup"] label[data-baseweb="radio"] span {
        color: #8b6f47 !important;
    }
    /* Obwódka przycisków */
    div.row-widget.stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        border: 2px solid #8b6f47 !important;
    }

     input[type="radio"] {
        accent-color: #8b6f47 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_NAME = "welcome_survey_clustering_pipeline_v1"

DATA = "welcome_survey_simple_v1.csv"

CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_descriptions_v1.json"


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)


@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters


with st.container(border=True):
    st.header("Powiedz nam coś o sobie")
    st.markdown("**Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania**")
    col7, col8 = st.columns(2)
    with col7:
        age = st.radio(
            "**Wiek:**",
            ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "\\>=65", "unknown"],
            horizontal=True,
        )
        edu_level = st.radio(
            "**Wykształcenie:**", ["Podstawowe", "Średnie", "Wyższe"], horizontal=True
        )

    with col8:
        fav_animals = st.radio(
            "**Ulubione zwierzęta:**",
            ["Psy", "Koty", "Koty i Psy", "Brak ulubionych", "Inne"],
            horizontal=True,
        )
        fav_place = st.radio(
            "**Ulubione miejsce:**",
            ["Nad wodą", "W lesie", "W górach", "Inne"],
            horizontal=True,
        )
    

   
    gender = st.pills("**Płeć:**", ["Mężczyzna", "Kobieta"], selection_mode="single")
    

person_df = pd.DataFrame(
    [
        {
            "age": age,
            "edu_level": edu_level,
            "fav_animals": fav_animals,
            "fav_place": fav_place,
            "gender": gender,
        }
    ]
)


model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

with st.container(border=True):

    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Twoja grupa to: ")
        st.markdown(
            f"<h3><span style='color:red'> {predicted_cluster_data['name']}</span></h3>",
            unsafe_allow_html=True,
        )

        st.markdown(f"###### {predicted_cluster_data['description']}")

    with col2:
        cluster_to_image = {
            "Cluster 0": "dog_forest_book.png",
            "Cluster 1": "dog_river.png",
            "Cluster 2": "dog_moutains.png",
            "Cluster 3": "cat_moutains.png",
            "Cluster 4": "dogs_cat_moutains.png",
            "Cluster 5": "dog_river.png",
            "Cluster 6": "dogs_cat_book_forest.png",
            "Cluster 7": "cats_river_book.png",
        }

        
        cluster_key = predicted_cluster_id
        #st.write(f"Klastr klucz: {cluster_key}")
        image_name = cluster_to_image.get(cluster_key)
        #st.write(f"Nazwa obrazka: {image_name}")

        if image_name:
            image_path = f"images/{image_name}"
            st.image(image_path, width=200)
        # else:
        #     st.warning(f"Brak przypisanego obrazka dla tego klastra ({cluster_key})")
        #     st.write("Dostępne klucze:", list(cluster_to_image.keys()))

        st.metric("Liczba znajomych", len(same_cluster_df))
        

    
    col3, col4 = st.columns(2, border=True)
    with col3:

        
        st.markdown(f"<h3>Rozkład wieku w grupie</h3>", unsafe_allow_html=True)

        fig = px.histogram(
            same_cluster_df.sort_values("age"),
            x="age",
            color_discrete_sequence=["#8b6f47"],
        )
        fig.update_layout(
            
            xaxis_title="Wiek",
            yaxis_title="Liczba osób",
            plot_bgcolor="#fff8ef",
            paper_bgcolor="#fff8ef",
        )
        st.plotly_chart(fig)

    with col4:
        st.markdown(f"<h3>Rozkład wykształcenia w grupie</h3>", unsafe_allow_html=True)
        fig = px.histogram(
            same_cluster_df, x="edu_level", color_discrete_sequence=["#8b6f47"]
        )
        fig.update_layout(
            
            xaxis_title="Wykształcenie",
            yaxis_title="Liczba osób",
            plot_bgcolor="#fff8ef",
            paper_bgcolor="#fff8ef",
        )
        st.plotly_chart(fig)

    col5, col6 = st.columns(2, border=True)

    with col5:
        st.markdown(
            f"<h3>Rozkład ulubionych zwierząt w grupie</h3>", unsafe_allow_html=True
        )
        fig = px.histogram(
            same_cluster_df, x="fav_animals", color_discrete_sequence=["#8b6f47"]
        )
        fig.update_layout(
            
            xaxis_title="Ulubione zwierzęta",
            yaxis_title="Liczba osób",
            plot_bgcolor="#fff8ef",
            paper_bgcolor="#fff8ef",
        )
        st.plotly_chart(fig)

    with col6:
        st.markdown(
            f"<h3>Rozkład ulubionych miejsc w grupie</h3>", unsafe_allow_html=True
        )
        fig = px.histogram(
            same_cluster_df, x="fav_place", color_discrete_sequence=["#8b6f47"]
        )
        fig.update_layout(
            
            xaxis_title="Ulubione miejsce",
            yaxis_title="Liczba osób",
            plot_bgcolor="#fff8ef",
            paper_bgcolor="#fff8ef",
        )
        st.plotly_chart(fig)

    with st.container(border=True):
        col9, colo10, col11 = st.columns([1, 2, 1])

        with colo10:
            st.markdown(f"<h3>Rozkład płci w grupie</h3>", unsafe_allow_html=True)
            fig = px.histogram(
                same_cluster_df, x="gender", color_discrete_sequence=["#8b6f47"]
            )
            fig.update_layout(
                
                xaxis_title="Płeć",
                yaxis_title="Liczba osób",
                plot_bgcolor="#fff8ef",
                paper_bgcolor="#fff8ef",
            )
            st.plotly_chart(fig, width="content")



