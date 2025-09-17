import os
import json
import warnings
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import joblib

warnings.filterwarnings("ignore")

# ========== CHARGEMENT DES DONNÉES ==========
from pathlib import Path

chemin_fichier = Path(__file__).resolve()         # Chemin complet du fichier .py
chemin_parent0= chemin_fichier.parents[0]         # Chemin dossier contenant le fichier
chemin_parent1= chemin_fichier.parents[1]


df = pd.read_csv(chemin_parent1/'Simulations'/'Data_processed'/'data_test_scaled.csv') # data_test_scaled (features traitées sans la cible)
data_train = pd.read_csv(chemin_parent1/'Simulations'/'Data_original'/'application_train.csv')  # Infos client + Pour comparaison
description = pd.read_csv(chemin_parent1/'Simulations'/'Data_original'/'HomeCredit_columns_description.csv',
                          usecols=['Row', 'Description'],
                          index_col=0,
                          encoding='unicode_escape')

ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df.columns if col not in ignore_features]


# ========== FONCTIONS UTILITAIRES ==========
@st.cache_data
def get_client_info(data, id_client):
    return data[data['SK_ID_CURR'] == int(id_client)]

@st.cache_data
def get_credit_decision(classe_predite):
    if classe_predite == 1:
        return "Crédit Refusé"
    else:
        return "Crédit Accordé"
        

@st.cache_data
def plot_distribution(applicationDF, feature, client_feature_val, title):
    if pd.isna(client_feature_val):
        st.warning("Valeur manquante pour ce client (NaN). Impossible de comparer.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    t0 = applicationDF[applicationDF['TARGET'] == 0]
    t1 = applicationDF[applicationDF['TARGET'] == 1]

    if feature == "DAYS_BIRTH":
        sns.kdeplot((t0[feature] / -365).dropna(), label='Remboursé', color='g', ax=ax)
        sns.kdeplot((t1[feature] / -365).dropna(), label='Défaillant', color='r', ax=ax)
        ax.axvline(float(client_feature_val / -365), color="blue", linestyle='--', label='Client')
        ax.set_xlabel("Âge (années)")
    elif feature == "DAYS_EMPLOYED":
        sns.kdeplot((t0[feature] / 365).dropna(), label='Remboursé', color='g', ax=ax)
        sns.kdeplot((t1[feature] / 365).dropna(), label='Défaillant', color='r', ax=ax)
        ax.axvline(float(client_feature_val / 365), color="blue", linestyle='--', label='Client')
        ax.set_xlabel("Ancienneté emploi (années)")
    else:
        sns.kdeplot(t0[feature].dropna(), label='Remboursé', color='g', ax=ax)
        sns.kdeplot(t1[feature].dropna(), label='Défaillant', color='r', ax=ax)
        ax.axvline(float(client_feature_val), color="blue", linestyle='--', label='Client')
        ax.set_xlabel(feature)

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.legend()
    st.pyplot(fig)

@st.cache_data
def univariate_categorical(applicationDF, feature, client_feature_val, title,
                           ylog=False, label_rotation=False, horizontal_layout=True):
    if pd.isna(client_feature_val.iloc[0]):
        st.warning("Valeur manquante pour ce client (NaN). Impossible de comparer.")
        return

    temp = applicationDF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Nombre': temp.values})

    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc["TARGET"] *= 100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    layout = (2, 1) if not horizontal_layout else (1, 2)
    fig, (ax1, ax2) = plt.subplots(*layout, figsize=(12, 5) if horizontal_layout else (15, 10))
    dynamic_hspace = min(1.5, 0.3 + 0.1*len(cat_perc))
    fig.subplots_adjust(hspace=dynamic_hspace) # espace vertical entre figures
    #fig.subplots_adjust(wspace=0.4) # espace horizontal entre figures

    sns.countplot(ax=ax1, x=feature, data=applicationDF,
                  hue="TARGET", order=cat_perc[feature],
                  palette=['g', 'r'])
    pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel("Nombre de clients")
    ax1.axvline(pos1, color="blue", linestyle='--', label='Client')
    if ylog:
        ax1.set_yscale('log')
    if label_rotation:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    sns.barplot(ax=ax2, x=feature, y='TARGET',
                order=cat_perc[feature], data=cat_perc, hue=feature, palette='Set2', legend=False)
    #sns.barplot(ax=ax2, x=feature, y='TARGET', data=cat_perc, color="skyblue")
    pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
    ax2.axvline(pos2, color="blue", linestyle='--', label='Client')
    if label_rotation:
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_title(f"{title} (% Défaillants)", fontsize=16)
    ax2.set_ylabel("% de défaillants par catégorie")

    st.pyplot(fig)


# ========== SIDEBAR ==========
import base64

# Charger l’image en base64
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convertir l’image
img_base64 = get_image_base64(chemin_parent0/"logo_entreprise.png")

# Injecter dans HTML
st.sidebar.markdown(
    f"""
    <div style="margin-top: -40px; text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

#st.sidebar.image("logo_entreprise.png", width=300)  # Remplace par ton logo ici

st.sidebar.title("Prêt à dépenser")
st.sidebar.markdown("**Analyse crédit - Dashboard**")

# Seuil de solvabilité à saiisr depuis la sidebar
seuil_solvabilite_str = st.sidebar.text_input(
    "Entrer le seuil de solvabilité [0,1]",
    value="0.5",
    help="Appuyez sur Entrée pour valider"
)

# Conversion et validation du seuil
try:
    seuil_solvabilite = float(seuil_solvabilite_str)
    if not (0.0 <= seuil_solvabilite <= 1.0):
        st.sidebar.error("La valeur doit être comprise entre 0.00 et 1.00")
        seuil_solvabilite = 0.5  # Valeur par défaut
except ValueError:
    st.sidebar.error("Veuillez entrer un nombre valide.")
    seuil_solvabilite = 0.5  # Valeur par défaut

    

id_list = df["SK_ID_CURR"].values
id_client = st.sidebar.selectbox("Sélectionner ID client", id_list)

with st.sidebar.expander("Options d'affichage", expanded=True):
    show_credit_decision = st.checkbox("Afficher la décision du modèle")
    show_client_details = st.checkbox("Afficher les infos client")
    show_client_comparison = st.checkbox("Comparer au reste des clients")
    shap_general = st.checkbox("Importance globale des variables")

with st.sidebar.expander("Aide sur les variables"):
    feature = st.selectbox("Choisir une variable", sorted(description.index.unique()))
    desc = description.loc[feature, 'Description']
    #st.markdown(f"**{desc}**")
    st.markdown(f"**{desc if isinstance(desc, str) else ', '.join(desc)}**")

# ========== HEADER PRINCIPAL ==========
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg, #0a4275, #3b8ed0);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .subheader {
        font-weight: 600;
        color: #0a4275;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    <div class="header">
        <h1>Dashboard de Scoring Crédit</h1>
        <p>Outil d'aide à la décision pour l’octroi de crédit à destination des gestionnaires</p>
    </div>
    """, unsafe_allow_html=True)

# ========== INFOS CLIENT ==========
client_info = get_client_info(data_train, id_client)
st.markdown(f"### Informations client - ID : {id_client}")

cols_map = {
    'CODE_GENDER': "GENRE", 'DAYS_BIRTH': "AGE", 'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
    'CNT_CHILDREN': "ENFANTS", 'FLAG_OWN_CAR': "VOITURE", 'FLAG_OWN_REALTY': "IMMOBILIER",
    'NAME_EDUCATION_TYPE': "ÉDUCATION", 'OCCUPATION_TYPE': "EMPLOI",
    'DAYS_EMPLOYED': "ANCIENNETÉ", 'AMT_INCOME_TOTAL': "REVENUS", 'AMT_CREDIT': "CRÉDIT",
    'NAME_CONTRACT_TYPE': "CONTRAT", 'AMT_ANNUITY': "ANNUITÉS", 'NAME_INCOME_TYPE': "TYPE REVENU",
    'EXT_SOURCE_1': "EXT1", 'EXT_SOURCE_2': "EXT2", 'EXT_SOURCE_3': "EXT3"
}

df_client = client_info[list(cols_map.keys())].rename(columns=cols_map)
df_client["AGE"] = (-df_client["AGE"] / 365).astype(int)
df_client["ANCIENNETÉ"] = (-df_client["ANCIENNETÉ"] / 365).astype(int)

if show_client_details:
    selected_cols = st.multiselect("Sélectionnez les infos à afficher :", options=df_client.columns,
                                   default=["GENRE", "AGE", "STATUT FAMILIAL", "REVENUS", "CRÉDIT"])
    st.table(df_client[selected_cols].T)

    if st.checkbox("Afficher toutes les données brutes du client"):
        st.dataframe(client_info)

# ========== DÉCISION DU MODÈLE ==========
if show_credit_decision:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Décision du modèle")

    try:
        #API_url = f"http://127.0.0.1:5000/credit/{id_client}"
        API_url = f"https://api-dep-emrr.onrender.com/credit/{id_client}"
        with st.spinner("Chargement des résultats du modèle..."):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            #prediction = API_data['prediction']
            proba = 1 - API_data['proba']
            # Prédiction binaire selon le seuil personnalisé
            prediction = int(proba > seuil_solvabilite)
            score = round(proba * 100, 2)

            col1, col2 = st.columns([1, 2])
            col1.metric("Risque de défaut", f"{score} %")

            #decision_text = "Crédit Accordé" if prediction == 0 else "Crédit Refusé"
            decision_text = get_credit_decision(classe_predite=prediction)
            color = "green" if prediction == 0 else "red"
            col1.markdown(f"<h3 style='color:{color}; font-weight:bold'>{decision_text}</h3>", unsafe_allow_html=True)

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "lightyellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': score}
                }
            ))
            col2.plotly_chart(gauge_fig, use_container_width=True)

        if st.checkbox("Voir les variables influençant la décision"):
            shap.initjs()
            X = df[df["SK_ID_CURR"] == int(id_client)][relevant_features]
            
            # Récupération des données depuis l'API
            shap_vals = np.array(API_data['shap_values_local'])
            base_value = API_data.get('base_value', 0)
            X_features = API_data['X_features']
            columns = X_features['columns']
            values = X_features['values'][0]
    
            X_df = pd.DataFrame([values], columns=columns)
    
            # Création de l'objet Explanation
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=X_df.values[0],
                feature_names=columns
            )
            
            plt.clf()
            plt.close('all')
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            fig = plt.gcf()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")

# ========== COMPARAISON CLIENT ==========
if show_client_comparison:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Comparaison avec les autres clients")

    

    # Sélection de la variable à comparer
    selected_var = st.selectbox("Variable à comparer", list(cols_map.values()))
    feature = [k for k, v in cols_map.items() if v == selected_var][0]
    
    # Détection automatique du type de variable
    if pd.api.types.is_numeric_dtype(data_train[feature]):
        # Si variable numérique (distribution)
        plot_distribution(
            data_train,
            feature,
            client_info[feature].values[0],
            selected_var
        )
    
    else:
        # Variable catégorielle (barplot)
    
        # Détection automatique : trop de modalités ?
        n_unique = data_train[feature].nunique()
        rotate_lbl = n_unique > 4       # Rotation si trop de catégories
        horizontal = n_unique < 6       # Affichage horizontal si beaucoup
    
        univariate_categorical(
            data_train,
            feature,
            client_info[feature],
            selected_var,
            ylog=False,
            label_rotation=rotate_lbl,
            horizontal_layout=horizontal
        )
       
    #API_url = f"http://127.0.0.1:5000/credit/{id_client}"
    API_url = f"https://api-dep-emrr.onrender.com/credit/{id_client}"
    with st.spinner("Chargement des résultats du modèle..."):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        
    X = df[df["SK_ID_CURR"] == int(id_client)][relevant_features]
            
    # Récupération des données depuis l'API
    shap_vals = np.array(API_data['shap_values_local'])
    base_value = API_data.get('base_value', 0)
    X_features = API_data['X_features']
    columns = X_features['columns']
    values = X_features['values'][0]

    X_df = pd.DataFrame([values], columns=columns)
    
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=X_df.values[0],
        feature_names=columns
    )
        
    # Créer un DataFrame combinant les noms de variables, les SHAP values et les valeurs du client
    shap_df = pd.DataFrame({
        "Feature": explanation.feature_names,
        "SHAP Value": explanation.values,
        "Feature Value": explanation.data
    })
    
    # Trier pour extraire les top 10 qui augmentent le risque (SHAP value > 0)
    top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    
    # Trier pour extraire les top 10 qui réduisent le risque (SHAP value < 0)
    top_negative_shap = shap_df.sort_values(by="SHAP Value").head(10)

    
    import plotly.graph_objects as go
    # Fonction pour générer un graphique Plotly horizontal
    def generate_shap_plot(df, title, color):
        # Trier par importance absolue
        df = df.reindex(df["SHAP Value"].abs().sort_values(ascending=True).index)
    
        # Labels combinés pour Y : nom + valeur de la variable
        y_labels = [f"{feat} ({val:.2f})" for feat, val in zip(df["Feature"], df["Feature Value"])]
    
        # Formater les SHAP values
        shap_values_text = [f"{val:+.2f}" for val in df["SHAP Value"]]
    
        fig = go.Figure(go.Bar(
            x=df["SHAP Value"],
            y=y_labels,
            orientation='h',
            marker=dict(color=color),
            text=shap_values_text,
            textposition='auto',  # auto: inside si assez de place, sinon outside
            #textangle=0,
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "SHAP: %{x:+.3f}<br>" +
                "Valeur: %{text}<extra></extra>"
            )
        ))
    
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
            ),
            xaxis_title="Valeur SHAP",
            yaxis_title="",
            width=900,    # largeur en pixels
            height=600,   # hauteur en pixels           
            yaxis=dict(
                automargin=True,
                tickfont=dict(size=11,color='black')
                
            ),
            margin=dict(l=270, r=10, t=40, b=20),
            showlegend=False,
        )
        return fig

    
    # Colonnes Streamlit
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            generate_shap_plot(
                top_positive_shap,
                "Top 10 - Augmentent le risque",
                color="crimson"
            ),
            use_container_width=True
        )
        
    
    with col2:
        st.plotly_chart(
            generate_shap_plot(
                top_negative_shap,
                "Top 10 - Réduisent le risque",
                color="green"
            ),
            use_container_width=True
        )

    def get_title_font_size(height):
        base_size = 12  # une taille de police de base
        scale_factor = height / 600.0  # supposons que 600 est la hauteur par défaut
        return base_size * scale_factor

    def find_closest_description(feature_name, definitions_df):
        for idx, row in definitions_df.iterrows():
            # idx correspond à 'Row'
            if idx in feature_name:
                return row["Description"]
        return None

    
    feature_names = columns
    feature_values = values
    definition_features_df=description

    def plot_distribution(selected_feature, col):

        
        if selected_feature:
            data = df[selected_feature]
    
            # Trouver la valeur de la fonctionnalité pour le client actuel :
            client_feature_value = feature_values[feature_names.index(selected_feature)]
    
            fig = go.Figure()
    
            # Vérifier si la fonctionnalité est catégorielle :
            unique_values = sorted(data.dropna().unique())
            if set(unique_values) <= {0, 1, 2, 3, 4, 5, 6, 7}:
                # Compter les occurrences de chaque valeur :
                counts = data.value_counts().sort_index()
    
                # Assurez-vous que les longueurs correspondent
                assert len(unique_values) == len(counts)
    
                # Modifier la déclaration de la liste de couleurs pour correspondre à la taille de unique_values
                colors = ["blue"] * len(unique_values)
    
                # Mettre à jour client_value
                client_value = (
                    unique_values.index(client_feature_value)
                    if client_feature_value in unique_values
                    else None
                )
    
                # Mettre à jour la couleur correspondante si client_value n'est pas None
                if client_value is not None:
                    colors[client_value] = "red"
    
                # Modifier le tracé pour utiliser unique_values
                fig.add_trace(go.Bar(x=unique_values, y=counts.values, marker_color=colors))
    
            else:
                # Calculer les bins pour le histogramme :
                hist_data, bins = np.histogram(data.dropna(), bins=20)
    
                # Trouvez le bin pour client_feature_value :
                client_bin_index = np.digitize(client_feature_value, bins) - 1
    
                # Créer une liste de couleurs pour les bins :
                colors = ["blue"] * len(hist_data)
                if (
                    0 <= client_bin_index < len(hist_data)
                ):  # Vérifiez que l'index est valide
                    colors[client_bin_index] = "red"
    
                # Tracer la distribution pour les variables continues :
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        marker=dict(color=colors, opacity=0.7),
                        name="Distribution",
                        xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                    )
                )
    
                # Utiliser une échelle logarithmique si la distribution est fortement asymétrique :
                mean_val = np.mean(hist_data)
                std_val = np.std(hist_data)
                if std_val > 3 * mean_val:  # Ce seuil peut être ajusté selon vos besoins
                    fig.update_layout(yaxis_type="log")
    
            height = 600  # Ajustez cette valeur selon la hauteur par défaut de votre figure ou obtenez-la d'une autre manière.
            title_font_size = get_title_font_size(height)
    
            fig.update_layout(
                title_text=f"Distribution pour {selected_feature}",
                title_font=dict(size=title_font_size),  # Ajoutez cette ligne
                xaxis_title=selected_feature,
                yaxis_title="Nombre de clients",
                title_x=0.3,
            )
    
            col.plotly_chart(fig, use_container_width=True)
    
            # Afficher la définition de la feature choisi :
            description = find_closest_description(selected_feature, definition_features_df)
            if description:
                col.write(f"**Definition:** {description}")


    # Créez des colonnes pour les listes déroulantes
    col1, col2 = st.columns(2)

    # Mettez la première liste déroulante dans col1
    with col1:
        selected_feature_positive = st.selectbox(
            "Sélectionnez une fonctionnalité augmentant le risque",
            [""] + top_positive_shap["Feature"].tolist(),
        )

    # Mettez la deuxième liste déroulante dans col2
    with col2:
        selected_feature_negative = st.selectbox(
            "Sélectionnez une fonctionnalité réduisant le risque",
            [""] + top_negative_shap["Feature"].tolist(),
        )
    plot_distribution(selected_feature_positive, col1)
    plot_distribution(selected_feature_negative, col2)

# ========== IMPORTANCE GLOBALE ==========
if shap_general:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Importance globale des variables")
    fig_path = chemin_parent0 / "global_feature_importance.png"
    st.image(str(fig_path), use_container_width=True)
