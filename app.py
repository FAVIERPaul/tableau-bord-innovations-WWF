import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

st.set_page_config(
    page_title="Tableau de bord des innovations",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Couleurs WWF
WWF_GREEN = "#00693E"
WWF_ORANGE = "#F07E16"
COLOR_LOW = "#ffcccc"      # Rouge clair
COLOR_MEDIUM = "#ffffcc"   # Jaune
COLOR_HIGH = "#ccffcc"     # Vert clair
COLOR_NA = "#f0f0f0"       # Gris

# --- FONCTION : Charger les données ---
@st.cache_data
def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name="Feuil1", header=1)
        df.columns = [str(c).strip() for c in df.columns]

        if "Innovation identifiée" not in df.columns:
            st.error("La colonne 'Innovation identifiée' est manquante dans le fichier Excel.")
            return pd.DataFrame()

        df = df.dropna(subset=["Innovation identifiée"])

        # Créer les labels de nature
        def get_nature_labels(row):
            labels = []
            if "Innovation technologique" in df.columns and row.get("Innovation technologique", 0) == 1:
                labels.append("Technologique")
            if "Innovation d'usage" in df.columns and row.get("Innovation d'usage", 0) == 1:
                labels.append("Usage")
            if "Innovation organisationnelle" in df.columns and row.get("Innovation organisationnelle", 0) == 1:
                labels.append("Organisationnelle")
            return labels if labels else ["Non classé"]

        df["Nature_labels"] = df.apply(get_nature_labels, axis=1)
        df["Nature_label"] = df["Nature_labels"].apply(lambda x: " + ".join(x) if x else "Non classé")

        return df

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return pd.DataFrame()

# --- FONCTION : Charger le dictionnaire ---
@st.cache_data
def load_dictionary():
    try:
        df_dict = pd.read_excel("Dictionnaire des variables.xlsx", sheet_name="Feuil1")
        return df_dict
    except Exception as e:
        st.error(f"Erreur lors du chargement du dictionnaire : {str(e)}")
        return pd.DataFrame()

# --- CHARGEMENT ---
excel_file = "Stats du rapport.xlsx"
df = load_data(excel_file)

if df.empty:
    st.warning("Aucune donnée n'a pu être chargée. Veuillez vérifier le fichier Excel.")
    st.stop()

# Colonnes de critères
colonnes_a_exclure = [
    "Innovation identifiée", "Innovation technologique", "Innovation d'usage",
    "Innovation organisationnelle", "Nature_label", "Nature_labels"
]
criteres = [c for c in df.columns if c not in colonnes_a_exclure]

# --- HEADER AVEC LOGOS ---
col_logo1, col_title, col_logo2 = st.columns([1, 3, 1])

with col_logo1:
    try:
        wwf_logo = Image.open("WWF_logo.png")
        st.image(wwf_logo, width=120)  # Réduit de 150 à 120
    except:
        pass

with col_title:
    st.title("Tableau de bord des innovations")
    st.markdown(f"""
    Cette visualisation présente **{len(df)} innovations** évaluées selon **{len(criteres)} critères**.
    Les scores vont de **0 (faible)** à **2 (fort)**.
    """)

with col_logo2:
    try:
        cems_logo = Image.open("CEMS_logo.png")
        st.image(cems_logo, width=200)  # Augmenté de 150 à 200
    except:
        pass

st.markdown("---")

# --- SIDEBAR : FILTRES ---
st.sidebar.header("Filtres")

# Filtre par nature
natures_disponibles = list(set([n for labels in df["Nature_labels"] for n in labels]))
natures_selectionnees = st.sidebar.multiselect(
    "Type d'innovation",
    options=sorted(natures_disponibles),
    default=sorted(natures_disponibles)
)

# Filtrer le dataframe
if natures_selectionnees:
    df_filtered = df[df["Nature_labels"].apply(lambda x: any(n in natures_selectionnees for n in x))]
else:
    df_filtered = df.copy()

st.sidebar.markdown(f"**{len(df_filtered)}** innovations affichées sur {len(df)}")

# --- ONGLETS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vue d'ensemble", "Comparaison", "Analyse par critère", "Données", "Dictionnaire"])

# ================== ONGLET 1: VUE D'ENSEMBLE ==================
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Répartition par type d'innovation")

        # Compter les innovations par type
        techno = len(df[df["Innovation technologique"] == 1])
        usage = len(df[df["Innovation d'usage"] == 1])
        orga = len(df[df["Innovation organisationnelle"] == 1])
        multi = len(df[(df["Innovation technologique"] == 1) &
                       ((df["Innovation d'usage"] == 1) | (df["Innovation organisationnelle"] == 1))])

        categories = ["Technologique", "Usage", "Organisationnelle"]
        values = [techno, usage, orga]
        colors = [WWF_GREEN, WWF_ORANGE, '#7A9D96']

        fig_venn = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )])

        fig_venn.update_layout(
            title="Nombre d'innovations par type",
            xaxis_title="Type d'innovation",
            yaxis_title="Nombre",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_venn, use_container_width=True)

        if multi > 0:
            st.info(f"**{multi}** innovation(s) appartiennent à plusieurs catégories")

        # Diagramme de Venn
        st.markdown("---")
        st.subheader("Diagramme de Venn des natures d'innovation")

        # Créer les ensembles
        set_techno = set(df[df["Innovation technologique"] == 1]["Innovation identifiée"])
        set_usage = set(df[df["Innovation d'usage"] == 1]["Innovation identifiée"])
        set_orga = set(df[df["Innovation organisationnelle"] == 1]["Innovation identifiée"])

        if set_techno or set_usage or set_orga:
            fig_venn_plot, ax = plt.subplots(figsize=(10, 8))

            # Créer le diagramme de Venn
            v = venn3([set_techno, set_usage, set_orga],
                     set_labels=('Technologique', 'Usage', 'Organisationnelle'),
                     set_colors=(WWF_GREEN, WWF_ORANGE, '#7A9D96'),
                     alpha=0.6,
                     ax=ax)

            # Améliorer l'apparence
            for text in v.set_labels:
                if text:
                    text.set_fontsize(12)
                    text.set_fontweight('bold')

            for text in v.subset_labels:
                if text:
                    text.set_fontsize(14)
                    text.set_fontweight('bold')

            plt.title("Intersections entre les types d'innovation", fontsize=14, fontweight='bold', pad=20)

            st.pyplot(fig_venn_plot)

            # Détails des intersections
            with st.expander("Voir le détail des intersections"):
                only_techno = set_techno - set_usage - set_orga
                only_usage = set_usage - set_techno - set_orga
                only_orga = set_orga - set_techno - set_usage
                techno_usage = (set_techno & set_usage) - set_orga
                techno_orga = (set_techno & set_orga) - set_usage
                usage_orga = (set_usage & set_orga) - set_techno
                all_three = set_techno & set_usage & set_orga

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**Technologique uniquement** ({len(only_techno)})")
                    for innov in sorted(only_techno):
                        st.markdown(f"- {innov}")

                    st.markdown(f"**Usage uniquement** ({len(only_usage)})")
                    for innov in sorted(only_usage):
                        st.markdown(f"- {innov}")

                    st.markdown(f"**Organisationnelle uniquement** ({len(only_orga)})")
                    for innov in sorted(only_orga):
                        st.markdown(f"- {innov}")

                with col_b:
                    st.markdown(f"**Technologique + Usage** ({len(techno_usage)})")
                    for innov in sorted(techno_usage):
                        st.markdown(f"- {innov}")

                    st.markdown(f"**Technologique + Organisationnelle** ({len(techno_orga)})")
                    for innov in sorted(techno_orga):
                        st.markdown(f"- {innov}")

                    st.markdown(f"**Usage + Organisationnelle** ({len(usage_orga)})")
                    for innov in sorted(usage_orga):
                        st.markdown(f"- {innov}")

                    st.markdown(f"**Les trois** ({len(all_three)})")
                    for innov in sorted(all_three):
                        st.markdown(f"- {innov}")

    with col2:
        st.subheader("Matrice de chaleur")

        # Créer la heatmap
        df_heatmap = df_filtered.set_index("Innovation identifiée")[criteres].fillna(-1)

        fig_heat = go.Figure(data=go.Heatmap(
            z=df_heatmap.values,
            x=[c.replace("/", "/<br>") for c in df_heatmap.columns],
            y=df_heatmap.index,
            colorscale=[
                [0, COLOR_NA],       # -1 (N/A) = gris clair
                [0.34, COLOR_LOW],   # 0 = rouge clair
                [0.67, COLOR_MEDIUM],# 1 = jaune
                [1, COLOR_HIGH]      # 2 = vert
            ],
            text=df_heatmap.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Score",
                tickvals=[-1, 0, 1, 2],
                ticktext=['N/A', 'Faible', 'Moyen', 'Fort']
            ),
            hovertemplate='<b>%{y}</b><br>%{x}<br>Score: %{z}<extra></extra>'
        ))

        fig_heat.update_layout(
            height=600,
            xaxis_tickangle=-45,
            margin=dict(l=200, r=20, t=40, b=150)
        )

        st.plotly_chart(fig_heat, use_container_width=True)

    # Statistiques globales
    st.subheader("Score moyen par critère")

    cols_stats = st.columns(len(criteres) if len(criteres) <= 4 else 4)

    for idx, critere in enumerate(criteres[:4]):
        with cols_stats[idx]:
            score_moyen = df_filtered[critere].mean()
            st.metric(
                label=critere.split("/")[0],
                value=f"{score_moyen:.2f}",
                delta=None
            )

# ================== ONGLET 2: COMPARAISON ==================
with tab2:
    st.subheader("Comparaison d'innovations")

    # Sélection des innovations à comparer
    innovations_a_comparer = st.multiselect(
        "Sélectionnez jusqu'à 5 innovations à comparer",
        options=df_filtered["Innovation identifiée"].tolist(),
        default=df_filtered["Innovation identifiée"].tolist()[:3],
        max_selections=5
    )

    if innovations_a_comparer:
        df_compare = df_filtered[df_filtered["Innovation identifiée"].isin(innovations_a_comparer)]

        # Radar chart
        fig_radar = go.Figure()

        # Couleurs pour chaque innovation
        innovation_colors = [WWF_GREEN, WWF_ORANGE, '#7A9D96', '#4A7C59', '#E89C3C']

        for idx, (_, row) in enumerate(df_compare.iterrows()):
            values = [row[c] for c in criteres]
            values_filled = [v if pd.notna(v) else 0 for v in values]

            fig_radar.add_trace(go.Scatterpolar(
                r=values_filled + [values_filled[0]],  # Fermer le polygone
                theta=criteres + [criteres[0]],
                fill='toself',
                name=row["Innovation identifiée"],
                line_color=innovation_colors[idx % len(innovation_colors)]
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2],
                    tickvals=[0, 1, 2],
                    ticktext=['Faible', 'Moyen', 'Fort']
                )
            ),
            showlegend=True,
            height=600,
            title="Comparaison multi-critères"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        # Tableau de comparaison
        st.markdown("### Tableau de comparaison détaillé")
        df_table = df_compare[["Innovation identifiée", "Nature_label"] + criteres]

        # Formater le tableau
        def color_score(val):
            if pd.isna(val):
                return f'background-color: {COLOR_NA}'
            elif val == 0:
                return f'background-color: {COLOR_LOW}'
            elif val == 1:
                return f'background-color: {COLOR_MEDIUM}'
            elif val == 2:
                return f'background-color: {COLOR_HIGH}'
            return ''

        styled_table = df_table.style.applymap(color_score, subset=criteres)
        st.dataframe(styled_table, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une innovation pour afficher la comparaison")

# ================== ONGLET 3: ANALYSE PAR CRITÈRE ==================
with tab3:
    st.subheader("Distribution par critère")

    critere_selectionne = st.selectbox("Choisissez un critère à analyser", criteres)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution des scores
        score_counts = df_filtered[critere_selectionne].value_counts().sort_index()

        fig_bar = go.Figure(data=[go.Bar(
            x=score_counts.index,
            y=score_counts.values,
            marker_color=[COLOR_LOW, COLOR_MEDIUM, COLOR_HIGH][:len(score_counts)],
            text=score_counts.values,
            textposition='auto',
        )])

        fig_bar.update_layout(
            title=f"Distribution des scores - {critere_selectionne}",
            xaxis_title="Score",
            yaxis_title="Nombre d'innovations",
            xaxis=dict(tickvals=[0, 1, 2], ticktext=['Faible', 'Moyen', 'Fort']),
            height=400
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Innovations par score
        st.markdown(f"**Innovations par niveau de score**")

        for score in sorted(df_filtered[critere_selectionne].dropna().unique()):
            innovations = df_filtered[df_filtered[critere_selectionne] == score]["Innovation identifiée"].tolist()
            label = ["Faible (0)", "Moyen (1)", "Fort (2)"][int(score)]

            with st.expander(f"**{label}** - {len(innovations)} innovation(s)"):
                for innov in innovations:
                    st.markdown(f"- {innov}")

# ================== ONGLET 4: DONNÉES ==================
with tab4:
    st.subheader("Tableau de données complet")

    # Options d'affichage
    col1, col2 = st.columns(2)
    with col1:
        show_all_cols = st.checkbox("Afficher toutes les colonnes", value=True)
    with col2:
        export_csv = st.button("Exporter en CSV")

    if export_csv:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name="innovations_filtrees.csv",
            mime="text/csv"
        )

    # Afficher le tableau
    if show_all_cols:
        st.dataframe(df_filtered, use_container_width=True, height=600)
    else:
        cols_to_show = ["Innovation identifiée", "Nature_label"] + criteres[:4]
        st.dataframe(df_filtered[cols_to_show], use_container_width=True, height=600)

    # Statistiques
    st.markdown("### Statistiques descriptives")
    st.dataframe(df_filtered[criteres].describe(), use_container_width=True)

# ================== ONGLET 5: DICTIONNAIRE ==================
with tab5:
    st.subheader("Dictionnaire des variables")

    df_dict = load_dictionary()

    if not df_dict.empty:
        st.markdown("""
        Ce dictionnaire présente la définition de chaque variable et les modalités de réponse.
        """)

        # Parser le dictionnaire pour l'afficher de manière structurée
        current_var = None
        current_note = None

        for idx, row in df_dict.iterrows():
            # Extraire les valeurs des colonnes
            col1_val = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            col2_val = row.iloc[2]
            col3_val = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""

            # Détecter une note (ligne 0)
            if col3_val and col3_val != "nan" and "Note sur les modalités" in col3_val:
                continue

            # Détecter une nouvelle variable (col1 rempli, col2 vide)
            if col1_val and col1_val != "nan" and pd.isna(col2_val):
                current_var = col1_val

                # Afficher la variable
                st.markdown(f"### {current_var}")

                # Afficher la description si disponible
                if col3_val and col3_val != "nan" and len(col3_val) > 0:
                    st.info(col3_val)
                    current_note = col3_val

            # Afficher les modalités (col1 rempli, col2 rempli)
            elif col1_val and col1_val != "nan" and pd.notna(col2_val):
                if current_var:
                    # C'est une modalité
                    val_display = int(col2_val) if isinstance(col2_val, float) and col2_val == int(col2_val) else col2_val
                    st.markdown(f"- **{col1_val}** : `{val_display}`")

        st.markdown("---")
        st.markdown("**Fichier source** : Dictionnaire des variables.xlsx")
    else:
        st.warning("Le dictionnaire des variables n'a pas pu être chargé.")

# --- FOOTER ---
st.sidebar.markdown("---")
if st.sidebar.button("Recharger les données"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("💡**Astuce** : Vous pouvez modifier le fichier Excel et cliquer sur 'Recharger les données' afin de mettre à jour les visualisations")
