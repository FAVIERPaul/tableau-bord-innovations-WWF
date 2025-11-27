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
WWF_TEAL = "#7A9D96"
COLOR_LOW = "#ffcccc"      # Rouge clair
COLOR_MEDIUM = "#ffffcc"   # Jaune
COLOR_HIGH = "#ccffcc"     # Vert clair
COLOR_NA = "#f0f0f0"       # Gris

# Fonction pour afficher un encart info avec texte en gras avant ":"
def info_bold(text):
    if ":" in text:
        parts = text.split(":", 1)
        formatted = f"<strong>{parts[0]}</strong> :{parts[1]}"
    else:
        formatted = text
    st.markdown(f"""
    <div style="background-color: #e8f4f8; border-left: 4px solid #31708f; padding: 12px 16px; border-radius: 4px; color: #31708f; margin: 8px 0;">
        {formatted}
    </div>
    """, unsafe_allow_html=True)

# Ordre d'apparition fixe
ORDRE_INTENSITE = ['Incrémentale', 'Radicale', 'Disruptive']
ORDRE_NATURE = ['Technologique', 'Usage', 'Organisationnelle']

# Mapping des couleurs par intensité
COULEURS_INTENSITE = {
    'Incrémentale': WWF_GREEN,
    'Radicale': WWF_ORANGE,
    'Disruptive': WWF_TEAL
}

# Mapping des couleurs par nature
COULEURS_NATURE = {
    'Technologique': WWF_GREEN,
    'Usage': WWF_ORANGE,
    'Organisationnelle': WWF_TEAL
}

# --- FONCTION : Charger les données ---
@st.cache_data
def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name="Statistique des critères", header=1)
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

        # Créer les labels d'intensité (incrémentale, radicale, disruptive)
        def get_intensite_labels(row):
            labels = []
            if "Innovation incrémentale" in df.columns and row.get("Innovation incrémentale", 0) == 1:
                labels.append("Incrémentale")
            if "Innovation radicale" in df.columns and row.get("Innovation radicale", 0) == 1:
                labels.append("Radicale")
            if "Innovation disruptive" in df.columns and row.get("Innovation disruptive", 0) == 1:
                labels.append("Disruptive")
            return labels if labels else ["Non classé"]

        df["Intensite_labels"] = df.apply(get_intensite_labels, axis=1)
        df["Intensite_label"] = df["Intensite_labels"].apply(lambda x: " + ".join(x) if x else "Non classé")

        return df

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return pd.DataFrame()

# --- FONCTION : Charger le dictionnaire ---
@st.cache_data
def load_dictionary():
    try:
        df_dict = pd.read_excel("Stats du rapport.xlsx", sheet_name="Dictionnaire des variables")
        return df_dict
    except Exception as e:
        st.error(f"Erreur lors du chargement du dictionnaire : {str(e)}")
        return pd.DataFrame()

# --- FONCTION : Extraire les max potentiels par critere depuis le dictionnaire ---
@st.cache_data
def get_max_par_critere():
    try:
        df_dict = pd.read_excel("Stats du rapport.xlsx", sheet_name="Dictionnaire des variables")
        current_var = None
        max_values = {}

        for idx, row in df_dict.iterrows():
            col1_val = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
            col2_val = row.iloc[2]

            # Detecter une nouvelle variable (col1 rempli, col2 vide)
            if col1_val and col1_val != 'nan' and pd.isna(col2_val):
                current_var = col1_val
                max_values[current_var] = 0

            # Detecter une modalite (col2 rempli avec un nombre)
            elif current_var and pd.notna(col2_val):
                try:
                    val = float(col2_val)
                    if val > max_values.get(current_var, 0):
                        max_values[current_var] = val
                except:
                    pass

        return max_values
    except Exception as e:
        return {}

# --- CHARGEMENT ---
excel_file = "Stats du rapport.xlsx"
df = load_data(excel_file)

if df.empty:
    st.warning("Aucune donnée n'a pu être chargée. Veuillez vérifier le fichier Excel.")
    st.stop()

# Colonnes de critères
colonnes_a_exclure = [
    "Innovation identifiée", "Innovation technologique", "Innovation d'usage",
    "Innovation organisationnelle", "Innovation incrémentale", "Innovation radicale",
    "Innovation disruptive", "Nature_label", "Nature_labels", "Intensite_label", "Intensite_labels"
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
    Les échelles de notation varient selon les critères.
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Vue d'ensemble", "Portefeuille d'innovations", "Comparaison", "Analyse par critère", "Données", "Dictionnaire", "Veille", "Retour d'expérience"])

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

        categories = ORDRE_NATURE
        values = [techno, usage, orga]
        colors = [COULEURS_NATURE[cat] for cat in categories]

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
                     set_labels=(ORDRE_NATURE[0], ORDRE_NATURE[1], ORDRE_NATURE[2]),
                     set_colors=(COULEURS_NATURE['Technologique'], COULEURS_NATURE['Usage'], COULEURS_NATURE['Organisationnelle']),
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
                    st.markdown(f"<span style='color:{COULEURS_NATURE['Technologique']}; font-weight:bold;'>Technologique uniquement</span> ({len(only_techno)})", unsafe_allow_html=True)
                    for innov in sorted(only_techno):
                        st.markdown(f"- {innov}")

                    st.markdown(f"<span style='color:{COULEURS_NATURE['Usage']}; font-weight:bold;'>Usage uniquement</span> ({len(only_usage)})", unsafe_allow_html=True)
                    for innov in sorted(only_usage):
                        st.markdown(f"- {innov}")

                    st.markdown(f"<span style='color:{COULEURS_NATURE['Organisationnelle']}; font-weight:bold;'>Organisationnelle uniquement</span> ({len(only_orga)})", unsafe_allow_html=True)
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
        st.subheader("Heatmap")

        # Creer la heatmap
        df_heatmap = df_filtered.set_index("Innovation identifiée")[criteres].fillna(-1)

        # Tronquer les noms de criteres pour l'affichage (max 25 caracteres)
        def truncate_label(label, max_len=25):
            if len(label) > max_len:
                return label[:max_len-2] + ".."
            return label

        labels_tronques = [truncate_label(c) for c in df_heatmap.columns]
        labels_complets = df_heatmap.columns.tolist()

        # Charger les max potentiels par critere depuis le dictionnaire
        max_par_critere = get_max_par_critere()

        # Creer une matrice normalisee (valeur / max du critere) pour les couleurs
        # et garder les valeurs originales pour l'affichage
        # Utiliser np.nan pour les valeurs N/A au lieu de -1
        df_heatmap_normalized = df_heatmap.copy()
        for col in df_heatmap.columns:
            max_crit = max_par_critere.get(col, df_heatmap[col].replace(-1, np.nan).max())
            if max_crit and max_crit > 0:
                df_heatmap_normalized[col] = df_heatmap[col].apply(
                    lambda x: x / max_crit if x >= 0 else np.nan
                )
            else:
                # Si pas de max, remplacer les -1 par nan
                df_heatmap_normalized[col] = df_heatmap[col].apply(
                    lambda x: x if x >= 0 else np.nan
                )

        # Creer une matrice de customdata avec le nom complet du critere et le max pour chaque cellule
        customdata_matrix = []
        for i in range(len(df_heatmap.index)):
            row_data = []
            for j, col in enumerate(labels_complets):
                max_crit = max_par_critere.get(col, '?')
                row_data.append(f"{col}<br>Max possible: {max_crit}")
            customdata_matrix.append(row_data)
        customdata_matrix = np.array(customdata_matrix)

        # Creer le texte d'affichage (valeurs originales, vide pour N/A)
        text_matrix = df_heatmap.values.copy().astype(object)
        text_matrix = np.where(text_matrix == -1, '', text_matrix)

        # Colorscale sobre (blanc/bleu tres clair -> rouge)
        # L'echelle va de 0 a 1, les valeurs NaN seront en gris clair
        colorscale = [
            [0, '#f7fbff'],       # 0% = blanc bleuté très clair
            [0.25, '#c6dbef'],    # 25% = bleu très clair
            [0.5, '#fcbba1'],     # 50% = rose saumon clair
            [0.75, '#fb6a4a'],    # 75% = rouge orangé
            [1, '#cb181d']        # 100% = rouge foncé
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=df_heatmap_normalized.values,
            x=labels_tronques,
            y=df_heatmap.index,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            text=text_matrix,  # Afficher les valeurs originales (vide pour N/A)
            texttemplate='%{text}',
            textfont={"size": 10, "color": "black"},
            hoverongaps=False,
            colorbar=dict(
                title="Intensité",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            ),
            customdata=customdata_matrix,
            hovertemplate='<b>%{y}</b><br>%{customdata}<br>Score: %{text}<extra></extra>'
        ))

        # Ajouter une trace invisible pour afficher les cellules N/A en gris
        # Creer un masque pour les valeurs N/A
        na_mask = df_heatmap.values == -1
        if na_mask.any():
            z_na = np.where(na_mask, 0.5, np.nan)
            fig_heat.add_trace(go.Heatmap(
                z=z_na,
                x=labels_tronques,
                y=df_heatmap.index,
                colorscale=[[0, '#e0e0e0'], [1, '#e0e0e0']],
                showscale=False,
                hoverinfo='skip'
            ))

        fig_heat.update_layout(
            height=600,
            xaxis=dict(
                tickangle=-60,
                tickfont=dict(size=10)
            ),
            margin=dict(l=200, r=20, t=40, b=180)
        )

        st.plotly_chart(fig_heat, use_container_width=True)

# ================== ONGLET 2: PORTEFEUILLE D'INNOVATIONS ==================
with tab2:
    st.subheader("Portefeuille d'innovations")
    st.markdown("""
    Cet onglet vous permet d'analyser **l'intensité de vos innovations**
    en fonction de leur classification : **incrémentale**, **radicale** ou **disruptive**.
    """)

    # ========== SECTION 1: ANALYSE PAR INTENSITÉ ==========
    st.markdown("### Répartition par intensité d'innovation")

    # Compter les innovations par intensité
    # Note: une innovation peut avoir plusieurs intensités
    df_incremental = df_filtered[df_filtered.get("Innovation incrémentale", pd.Series([0]*len(df_filtered))) == 1]
    df_radical = df_filtered[df_filtered.get("Innovation radicale", pd.Series([0]*len(df_filtered))) == 1]
    df_disruptive = df_filtered[df_filtered.get("Innovation disruptive", pd.Series([0]*len(df_filtered))) == 1]

    count_incremental = len(df_incremental)
    count_radical = len(df_radical)
    count_disruptive = len(df_disruptive)

    # Nombre total d'innovations
    total_innovations = len(df_filtered)

    # Créer deux colonnes pour les visualisations
    col1, col2 = st.columns([1, 1])

    with col1:
        # Donut chart de l'intensité
        if count_incremental + count_radical + count_disruptive > 0:
            fig_donut = go.Figure(data=[go.Pie(
                labels=ORDRE_INTENSITE,
                values=[count_incremental, count_radical, count_disruptive],
                hole=0.5,
                marker=dict(colors=[COULEURS_INTENSITE[i] for i in ORDRE_INTENSITE]),
                textinfo='label+percent',
                textposition='outside',
                hovertemplate='<b>%{label}</b><br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>'
            )])

            fig_donut.update_layout(
                title={
                    'text': f"Distribution des intensités<br><sub>Sur {total_innovations} innovations</sub>",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=450,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    traceorder='normal'
                )
            )

            # Forcer l'ordre de la légende
            fig_donut.data[0].sort = False

            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Aucune innovation n'a été classifiée par intensité.")

    with col2:
        # Bar chart avec pourcentages
        if count_incremental + count_radical + count_disruptive > 0:
            total_count = count_incremental + count_radical + count_disruptive
            pct_incremental = (count_incremental / total_count * 100) if total_count > 0 else 0
            pct_radical = (count_radical / total_count * 100) if total_count > 0 else 0
            pct_disruptive = (count_disruptive / total_count * 100) if total_count > 0 else 0

            fig_bar = go.Figure(data=[go.Bar(
                x=ORDRE_INTENSITE,
                y=[count_incremental, count_radical, count_disruptive],
                text=[f'{count_incremental}<br>({pct_incremental:.1f}%)',
                      f'{count_radical}<br>({pct_radical:.1f}%)',
                      f'{count_disruptive}<br>({pct_disruptive:.1f}%)'],
                textposition='auto',
                marker=dict(color=[COULEURS_INTENSITE[i] for i in ORDRE_INTENSITE]),
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<extra></extra>'
            )])

            fig_bar.update_layout(
                title="Nombre d'innovations par intensité",
                xaxis_title="Type d'intensité",
                yaxis_title="Nombre d'innovations",
                height=450,
                showlegend=False
            )

            st.plotly_chart(fig_bar, use_container_width=True)

    # Métriques clés
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            label="Total innovations",
            value=total_innovations,
            delta=None
        )

    with col_m2:
        pct_incr = (count_incremental / total_innovations * 100) if total_innovations > 0 else 0
        st.metric(
            label="Incrémentales",
            value=count_incremental,
            delta=f"{pct_incr:.0f}% du portefeuille"
        )

    with col_m3:
        pct_rad = (count_radical / total_innovations * 100) if total_innovations > 0 else 0
        st.metric(
            label="Radicales",
            value=count_radical,
            delta=f"{pct_rad:.0f}% du portefeuille"
        )

    with col_m4:
        pct_dis = (count_disruptive / total_innovations * 100) if total_innovations > 0 else 0
        st.metric(
            label="Disruptives",
            value=count_disruptive,
            delta=f"{pct_dis:.0f}% du portefeuille"
        )

    # Note explicative
    st.info("**Note** : Une innovation peut appartenir à plusieurs catégories d'intensité. Les pourcentages sont calculés sur le total de {0} innovations.".format(total_innovations))

    # Liste détaillée des innovations par intensité (dans des expanders)
    with st.expander("Voir le détail des innovations par intensité"):
        col_list1, col_list2, col_list3 = st.columns(3)

        with col_list1:
            st.markdown(f"<span style='color:{COULEURS_INTENSITE['Incrémentale']}; font-weight:bold;'>Incrémentales</span> ({count_incremental})", unsafe_allow_html=True)
            incremental_list = df_incremental["Innovation identifiée"].tolist()
            if incremental_list:
                for innov in sorted(incremental_list):
                    st.markdown(f"- {innov}")
            else:
                st.info("Aucune innovation incrémentale")

        with col_list2:
            st.markdown(f"<span style='color:{COULEURS_INTENSITE['Radicale']}; font-weight:bold;'>Radicales</span> ({count_radical})", unsafe_allow_html=True)
            radical_list = df_radical["Innovation identifiée"].tolist()
            if radical_list:
                for innov in sorted(radical_list):
                    st.markdown(f"- {innov}")
            else:
                st.info("Aucune innovation radicale")

        with col_list3:
            st.markdown(f"<span style='color:{COULEURS_INTENSITE['Disruptive']}; font-weight:bold;'>Disruptives</span> ({count_disruptive})", unsafe_allow_html=True)
            disruptive_list = df_disruptive["Innovation identifiée"].tolist()
            if disruptive_list:
                for innov in sorted(disruptive_list):
                    st.markdown(f"- {innov}")
            else:
                st.info("Aucune innovation disruptive")

    # ========== SECTION 2: CROISEMENT NATURE x INTENSITÉ ==========
    st.markdown("---")
    st.markdown("### Croisement Nature × Intensité")

    st.markdown("Choisissez une visualisation :")

    viz_type = st.radio(
        "Type de visualisation",
        ["Graphique en barres groupées", "Heatmap", "Tableau croisé"],
        horizontal=True
    )

    # Créer un DataFrame pour le croisement
    cross_data = []
    for _, row in df_filtered.iterrows():
        innov_name = row["Innovation identifiée"]
        # Pour chaque innovation, créer une ligne pour chaque combinaison Nature x Intensité
        natures = row.get("Nature_labels", ["Non classé"])

        is_incr = row.get("Innovation incrémentale", 0) == 1
        is_rad = row.get("Innovation radicale", 0) == 1
        is_disr = row.get("Innovation disruptive", 0) == 1

        for nature in natures:
            if is_incr:
                cross_data.append({"Nature": nature, "Intensité": "Incrémentale", "Innovation": innov_name})
            if is_rad:
                cross_data.append({"Nature": nature, "Intensité": "Radicale", "Innovation": innov_name})
            if is_disr:
                cross_data.append({"Nature": nature, "Intensité": "Disruptive", "Innovation": innov_name})

    df_cross = pd.DataFrame(cross_data)

    if not df_cross.empty and "Intensité" in df_cross.columns:
        if viz_type == "Graphique en barres groupées":
            # Compter les combinaisons
            cross_counts = df_cross.groupby(['Nature', 'Intensité']).size().reset_index(name='count')

            # Assurer l'ordre correct pour les natures et intensités
            cross_counts['Nature'] = pd.Categorical(cross_counts['Nature'], categories=ORDRE_NATURE, ordered=True)
            cross_counts['Intensité'] = pd.Categorical(cross_counts['Intensité'], categories=ORDRE_INTENSITE, ordered=True)
            cross_counts = cross_counts.sort_values(['Nature', 'Intensité'])

            fig_grouped = px.bar(
                cross_counts,
                x='Nature',
                y='count',
                color='Intensité',
                barmode='group',
                title="Répartition des intensités par nature d'innovation",
                color_discrete_map=COULEURS_INTENSITE,
                labels={'count': 'Nombre d\'innovations', 'Nature': 'Nature de l\'innovation'},
                category_orders={'Nature': ORDRE_NATURE, 'Intensité': ORDRE_INTENSITE}
            )

            fig_grouped.update_layout(height=500)
            st.plotly_chart(fig_grouped, use_container_width=True)

        elif viz_type == "Heatmap":
            # Créer un pivot table pour la heatmap
            pivot_for_heat = df_cross.groupby(['Nature', 'Intensité']).size().reset_index(name='count')
            pivot_matrix = pivot_for_heat.pivot(index='Nature', columns='Intensité', values='count').fillna(0)

            # Réordonner selon l'ordre fixe
            pivot_matrix = pivot_matrix.reindex(index=ORDRE_NATURE, columns=ORDRE_INTENSITE, fill_value=0)

            fig_heat_cross = go.Figure(data=go.Heatmap(
                z=pivot_matrix.values,
                x=pivot_matrix.columns,
                y=pivot_matrix.index,
                colorscale=[[0, '#ffffff'], [0.5, WWF_ORANGE], [1, WWF_GREEN]],
                text=pivot_matrix.values,
                texttemplate='%{text}',
                textfont={"size": 14},
                hovertemplate='Nature: %{y}<br>Intensité: %{x}<br>Count: %{z}<extra></extra>'
            ))

            fig_heat_cross.update_layout(
                title="Heatmap Nature × Intensité",
                xaxis_title="Intensité",
                yaxis_title="Nature",
                height=500
            )

            st.plotly_chart(fig_heat_cross, use_container_width=True)

        elif viz_type == "Tableau croisé":
            # Tableau croisé avec ordre fixe
            pivot_table = pd.crosstab(
                df_cross['Nature'],
                df_cross['Intensité'],
                margins=True,
                margins_name="Total"
            )

            # Réordonner
            pivot_table = pivot_table.reindex(index=ORDRE_NATURE + ['Total'], columns=ORDRE_INTENSITE + ['Total'], fill_value=0)

            st.dataframe(pivot_table, use_container_width=True)
    else:
        st.warning("Aucune donnée disponible pour le croisement Nature × Intensité")        


# ================== ONGLET 3: COMPARAISON ==================
with tab3:
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

        # Recuperer les max par critere depuis le dictionnaire
        max_par_critere_radar = get_max_par_critere()

        # Calculer les max effectifs pour chaque critere (dictionnaire ou max observe)
        max_par_critere_effectif = {}
        for c in criteres:
            max_dict = max_par_critere_radar.get(c, None)
            max_obs = df_filtered[c].max()
            if max_dict and max_dict > 0:
                max_par_critere_effectif[c] = max_dict
            elif pd.notna(max_obs) and max_obs > 0:
                max_par_critere_effectif[c] = max_obs
            else:
                max_par_critere_effectif[c] = 1  # Eviter division par zero

        # Tronquer les noms de criteres pour l'affichage
        def truncate_label_radar(label, max_len=20):
            if len(label) > max_len:
                return label[:max_len-2] + ".."
            return label

        criteres_tronques = [truncate_label_radar(c) for c in criteres]

        for idx, (_, row) in enumerate(df_compare.iterrows()):
            # Valeurs originales
            values_orig = [row[c] for c in criteres]
            # Valeurs normalisees (0 a 100%) par rapport au max de chaque critere
            values_normalized = []
            for i, c in enumerate(criteres):
                val = values_orig[i]
                if pd.notna(val):
                    values_normalized.append((val / max_par_critere_effectif[c]) * 100)
                else:
                    values_normalized.append(0)

            # Texte pour le hover (valeur originale / max)
            hover_texts = []
            for i, c in enumerate(criteres):
                val = values_orig[i]
                max_c = max_par_critere_effectif[c]
                if pd.notna(val):
                    hover_texts.append(f"{c}<br>Score: {int(val)}/{int(max_c)} ({values_normalized[i]:.0f}%)")
                else:
                    hover_texts.append(f"{c}<br>Score: N/A")

            fig_radar.add_trace(go.Scatterpolar(
                r=values_normalized + [values_normalized[0]],  # Fermer le polygone
                theta=criteres_tronques + [criteres_tronques[0]],
                fill='toself',
                name=row["Innovation identifiée"],
                line_color=innovation_colors[idx % len(innovation_colors)],
                customdata=hover_texts + [hover_texts[0]],
                hovertemplate='<b>%{customdata}</b><extra></extra>'
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                )
            ),
            showlegend=True,
            height=600,
            title="Comparaison multi-critères (normalisée par critère)"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        st.caption("Chaque axe est normalisé par rapport au maximum possible de son critère (0% = 0, 100% = max du critère)")

        # Tableau de comparaison
        st.markdown("### Tableau de comparaison détaillé")
        df_table = df_compare[["Innovation identifiée", "Nature_label"] + criteres]

        # Formater le tableau avec couleurs dynamiques
        max_val_table = df_filtered[criteres].max().max()

        def color_score(val):
            if pd.isna(val):
                return f'background-color: {COLOR_NA}'
            if max_val_table == 0:
                return f'background-color: {COLOR_MEDIUM}'
            ratio = val / max_val_table
            if ratio < 0.33:
                return f'background-color: {COLOR_LOW}'
            elif ratio < 0.67:
                return f'background-color: {COLOR_MEDIUM}'
            else:
                return f'background-color: {COLOR_HIGH}'

        styled_table = df_table.style.applymap(color_score, subset=criteres)
        st.dataframe(styled_table, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une innovation pour afficher la comparaison")

# ================== ONGLET 4: ANALYSE PAR CRITÈRE ==================
with tab4:
    st.subheader("Distribution par critère")

    critere_selectionne = st.selectbox("Choisissez un critère à analyser", criteres)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution des scores
        score_counts = df_filtered[critere_selectionne].value_counts().sort_index()
        scores_uniques = sorted(score_counts.index)
        val_min_crit = min(scores_uniques)
        val_max_crit = max(scores_uniques)

        # Creer un gradient de couleurs adapte au nombre de valeurs
        def get_color_for_score(score, min_val, max_val):
            if max_val == min_val:
                return COLOR_MEDIUM
            ratio = (score - min_val) / (max_val - min_val)
            if ratio < 0.33:
                return COLOR_LOW
            elif ratio < 0.67:
                return COLOR_MEDIUM
            else:
                return COLOR_HIGH

        bar_colors = [get_color_for_score(s, val_min_crit, val_max_crit) for s in score_counts.index]

        fig_bar = go.Figure(data=[go.Bar(
            x=score_counts.index,
            y=score_counts.values,
            marker_color=bar_colors,
            text=score_counts.values,
            textposition='auto',
        )])

        fig_bar.update_layout(
            title=f"Distribution des scores - {critere_selectionne}",
            xaxis_title="Score",
            yaxis_title="Nombre d'innovations",
            xaxis=dict(tickvals=scores_uniques),
            height=400
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Innovations par score
        st.markdown(f"**Innovations par niveau de score**")

        scores_tries = sorted(df_filtered[critere_selectionne].dropna().unique())
        for score in scores_tries:
            innovations = df_filtered[df_filtered[critere_selectionne] == score]["Innovation identifiée"].tolist()
            label = f"Score {int(score)}"

            with st.expander(f"**{label}** - {len(innovations)} innovation(s)"):
                for innov in innovations:
                    st.markdown(f"- {innov}")

    # Statistiques globales avec mini-échelles
    st.subheader("Score moyen par critère")

    with st.expander("Comment lire ces scores ?"):
        st.markdown("""
        **Calcul du score moyen :**
        - Chaque innovation est évaluée sur plusieurs critères
        - Le score moyen correspond à la **moyenne arithmétique** des scores attribués à toutes les innovations pour ce critère
        - L'échelle de notation varie selon les critères (ex: 0-2 ou 0-1)

        **Lecture de l'échelle :**
        - Le curseur noir indique la position de la moyenne sur l'échelle du critère
        - Les valeurs min et max affichées correspondent aux bornes réelles du critère
        """)

    cols_stats = st.columns(len(criteres) if len(criteres) <= 4 else 4)

    for idx, critere in enumerate(criteres[:4]):
        with cols_stats[idx]:
            values = df_filtered[critere].dropna()
            score_moyen = values.mean()
            val_min = int(values.min())
            val_max = int(values.max())

            st.metric(
                label=critere.split("/")[0],
                value=f"{score_moyen:.2f}",
                delta=None
            )

            # Mini-echelle adaptee au min/max du critere
            if val_max > val_min:
                pct = ((score_moyen - val_min) / (val_max - val_min)) * 100
            else:
                pct = 50

            # Graduation du milieu
            val_mid = (val_min + val_max) / 2

            st.markdown(f"""
            <div style="position:relative; height:25px; margin-top:-10px;">
                <div style="background:#e0e0e0; height:6px; border-radius:3px; position:relative;">
                    <div style="background:{COLOR_LOW}; width:33%; height:100%; position:absolute; left:0; border-radius:3px 0 0 3px;"></div>
                    <div style="background:{COLOR_MEDIUM}; width:34%; height:100%; position:absolute; left:33%;"></div>
                    <div style="background:{COLOR_HIGH}; width:33%; height:100%; position:absolute; right:0; border-radius:0 3px 3px 0;"></div>
                    <div style="background:#333; width:8px; height:12px; border-radius:2px; position:absolute; top:-3px; left:calc({pct}% - 4px);"></div>
                </div>
                <div style="position:relative; font-size:9px; color:#666; margin-top:2px; height:12px;">
                    <span style="position:absolute; left:0;">{val_min}</span>
                    <span style="position:absolute; left:50%; transform:translateX(-50%);">{val_mid:g}</span>
                    <span style="position:absolute; right:0;">{val_max}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================== ONGLET 5: DONNÉES ==================
with tab5:
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

# ================== ONGLET 6: DICTIONNAIRE ==================
with tab6:
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
                    info_bold(col3_val)
                    current_note = col3_val

            # Afficher les modalités (col1 rempli, col2 rempli)
            elif col1_val and col1_val != "nan" and pd.notna(col2_val):
                if current_var:
                    # C'est une modalité
                    val_display = int(col2_val) if isinstance(col2_val, float) and col2_val == int(col2_val) else col2_val
                    st.markdown(f"- **{col1_val}** : `{val_display}`")

        st.markdown("---")
        st.markdown("**Fichier source** : Stats du rapport.xlsx (feuille 'Dictionnaire des variables')")
    else:
        st.warning("Le dictionnaire des variables n'a pas pu être chargé.")

# ================== ONGLET 7: VEILLE ==================
with tab7:
    st.subheader("Veille et ressources")

    st.markdown("""
    Cet onglet regroupe les **ressources externes** utiles pour la veille sur l'innovation
    et les données environnementales. Cliquez sur les cartes pour accéder aux sources.
    """)

    # Fichiers CSV
    fichier_ressources = "ressources_validees.csv"
    fichier_suggestions = "suggestions_veille.csv"
    import os
    from datetime import datetime

    # Palette de couleurs pour les cartes (alternance)
    COULEURS_CARTES = [
        {"gradient": "linear-gradient(135deg, #00693E 0%, #4A7C59 100%)", "btn": "#00693E"},
        {"gradient": "linear-gradient(135deg, #2E7D9A 0%, #7A9D96 100%)", "btn": "#2E7D9A"},
        {"gradient": "linear-gradient(135deg, #F07E16 0%, #E89C3C 100%)", "btn": "#F07E16"},
        {"gradient": "linear-gradient(135deg, #8B4513 0%, #A0522D 100%)", "btn": "#8B4513"},
    ]

    # Initialiser le fichier des ressources validées s'il n'existe pas
    if not os.path.exists(fichier_ressources):
        ressources_initiales = pd.DataFrame({
            'Nom': ['Innovation Example Repository - WWF', 'Catalogue - OFB'],
            'URL': [
                'https://sites.google.com/impacthub.net/innovation-wwf/our-impact/stories-repository?authuser=0',
                'https://data.ofb.fr/catalogue/Donnees-geographiques-OFB/fre/catalog.search#/home'
            ],
            'Description': [
                'Plateforme Innovation Hub présentant des innovations du réseau WWF. Découvrez des cas concrets d\'innovations mises en œuvre pour la conservation.',
                'Catalogue de données géographiques de l\'Office Français de la Biodiversité (OFB). Accès aux données environnementales et cartographiques nationales.'
            ],
            'Source': ['Impact Hub / WWF', 'Office Français de la Biodiversité'],
            'Date_validation': [datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")]
        })
        ressources_initiales.to_csv(fichier_ressources, index=False)

    # Fonction pour générer une carte de ressource
    def afficher_carte_ressource(nom, url, description, source, couleur_idx):
        couleur = COULEURS_CARTES[couleur_idx % len(COULEURS_CARTES)]
        st.markdown(f"""
        <div style="
            background: {couleur['gradient']};
            padding: 25px;
            border-radius: 15px;
            color: white;
            min-height: 250px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        ">
            <h3 style="margin-top:0; color:white; font-size: 18px;">{nom}</h3>
            <p style="font-size: 14px; line-height: 1.6;">
                {description}
            </p>
            <p style="font-size: 12px; opacity: 0.8; margin-bottom: 15px;">
                Source : {source}
            </p>
            <a href="{url}"
               target="_blank"
               style="
                   background-color: white;
                   color: {couleur['btn']};
                   padding: 10px 20px;
                   border-radius: 25px;
                   text-decoration: none;
                   font-weight: bold;
                   display: inline-block;
               ">
                Accéder à la ressource
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section des ressources validées
    st.markdown("### Ressources disponibles")

    # Charger et afficher les ressources validées
    if os.path.exists(fichier_ressources):
        df_ressources = pd.read_csv(fichier_ressources)

        if not df_ressources.empty:
            # Afficher les cartes en grille de 2 colonnes
            cols = st.columns(2)
            for idx, row in df_ressources.iterrows():
                with cols[idx % 2]:
                    afficher_carte_ressource(
                        row['Nom'],
                        row['URL'],
                        row['Description'],
                        row['Source'],
                        idx
                    )
    else:
        st.info("Aucune ressource disponible pour le moment.")

    st.markdown("---")

    # Section pour suggérer des ressources
    st.markdown("### Suggérer une ressource")

    with st.form("formulaire_suggestion", clear_on_submit=True):
        nom_ressource = st.text_input("Nom de la ressource *", placeholder="Ex: Portail de données biodiversité")
        url_ressource = st.text_input("URL *", placeholder="https://...")
        description_ressource = st.text_area("Description", placeholder="Décrivez brièvement cette ressource...")
        source_ressource = st.text_input("Source / Organisme", placeholder="Ex: Ministère de l'Environnement")

        submitted = st.form_submit_button("Envoyer la suggestion")

        if submitted:
            if nom_ressource and url_ressource:
                nouvelle_suggestion = pd.DataFrame({
                    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Nom': [nom_ressource],
                    'URL': [url_ressource],
                    'Description': [description_ressource if description_ressource else ""],
                    'Source': [source_ressource if source_ressource else ""]
                })

                if os.path.exists(fichier_suggestions):
                    df_suggestions = pd.read_csv(fichier_suggestions)
                    df_suggestions = pd.concat([df_suggestions, nouvelle_suggestion], ignore_index=True)
                else:
                    df_suggestions = nouvelle_suggestion

                df_suggestions.to_csv(fichier_suggestions, index=False)
                st.success("Merci ! Votre suggestion a été enregistrée et sera examinée par l'administrateur.")
            else:
                st.error("Veuillez remplir au minimum le nom et l'URL de la ressource.")

    # Section Administration (pour valider les suggestions)
    st.markdown("---")
    with st.expander("🔧 Administration des ressources"):
        st.markdown("#### Suggestions en attente de validation")

        if os.path.exists(fichier_suggestions):
            df_suggestions = pd.read_csv(fichier_suggestions)

            if not df_suggestions.empty:
                st.dataframe(df_suggestions, use_container_width=True, hide_index=True)
                st.caption(f"{len(df_suggestions)} suggestion(s) en attente")

                st.markdown("---")
                st.markdown("#### Valider une suggestion")

                # Sélection de la suggestion à valider
                suggestions_noms = df_suggestions['Nom'].tolist()
                suggestion_a_valider = st.selectbox(
                    "Sélectionnez la suggestion à valider",
                    options=[""] + suggestions_noms,
                    key="select_suggestion"
                )

                col_btn1, col_btn2 = st.columns(2)

                with col_btn1:
                    if st.button("✅ Valider cette suggestion", type="primary", disabled=not suggestion_a_valider):
                        if suggestion_a_valider:
                            # Récupérer les données de la suggestion
                            suggestion_data = df_suggestions[df_suggestions['Nom'] == suggestion_a_valider].iloc[0]

                            # Ajouter aux ressources validées
                            nouvelle_ressource = pd.DataFrame({
                                'Nom': [suggestion_data['Nom']],
                                'URL': [suggestion_data['URL']],
                                'Description': [suggestion_data['Description'] if pd.notna(suggestion_data['Description']) else ""],
                                'Source': [suggestion_data['Source'] if pd.notna(suggestion_data['Source']) else ""],
                                'Date_validation': [datetime.now().strftime("%Y-%m-%d")]
                            })

                            if os.path.exists(fichier_ressources):
                                df_ressources_val = pd.read_csv(fichier_ressources)
                                df_ressources_val = pd.concat([df_ressources_val, nouvelle_ressource], ignore_index=True)
                            else:
                                df_ressources_val = nouvelle_ressource

                            df_ressources_val.to_csv(fichier_ressources, index=False)

                            # Supprimer des suggestions
                            df_suggestions = df_suggestions[df_suggestions['Nom'] != suggestion_a_valider]
                            df_suggestions.to_csv(fichier_suggestions, index=False)

                            st.success(f"✅ '{suggestion_a_valider}' a été validée et ajoutée aux ressources !")
                            st.rerun()

                with col_btn2:
                    if st.button("❌ Rejeter cette suggestion", disabled=not suggestion_a_valider):
                        if suggestion_a_valider:
                            df_suggestions = df_suggestions[df_suggestions['Nom'] != suggestion_a_valider]
                            df_suggestions.to_csv(fichier_suggestions, index=False)
                            st.warning(f"'{suggestion_a_valider}' a été rejetée.")
                            st.rerun()
            else:
                st.info("Aucune suggestion en attente.")
        else:
            st.info("Aucune suggestion en attente.")

        st.markdown("---")
        st.markdown("#### Gérer les ressources existantes")

        if os.path.exists(fichier_ressources):
            df_ressources_admin = pd.read_csv(fichier_ressources)
            if not df_ressources_admin.empty:
                ressource_a_supprimer = st.selectbox(
                    "Sélectionnez une ressource à supprimer",
                    options=[""] + df_ressources_admin['Nom'].tolist(),
                    key="select_ressource_suppr"
                )

                if st.button("🗑️ Supprimer cette ressource", disabled=not ressource_a_supprimer):
                    if ressource_a_supprimer:
                        df_ressources_admin = df_ressources_admin[df_ressources_admin['Nom'] != ressource_a_supprimer]
                        df_ressources_admin.to_csv(fichier_ressources, index=False)
                        st.warning(f"'{ressource_a_supprimer}' a été supprimée.")
                        st.rerun()

    # Résumé des ressources
    st.markdown("---")
    st.markdown("### Résumé des ressources")

    if os.path.exists(fichier_ressources):
        df_resume = pd.read_csv(fichier_ressources)[['Nom', 'Source', 'Date_validation']]
        df_resume.columns = ['Ressource', 'Organisme', 'Date ajout']
        st.dataframe(df_resume, use_container_width=True, hide_index=True)

# ================== ONGLET 8: RETOUR D'EXPÉRIENCE ==================
with tab8:
    st.subheader("Partager un retour d'expérience")

    st.markdown("""
    Cet espace vous permet de **partager votre expérience** sur une innovation identifiée.
    Vos retours sont précieux pour enrichir la connaissance collective et aider d'autres programmes
    à bénéficier de vos apprentissages.
    """)

    st.markdown("---")

    # Liste des programmes WWF
    PROGRAMMES_WWF = [
        "Eau douce",
        "Vie sauvage",
        "Océans",
        "Forêts",
        "Agriculture et alimentation",
        "Partenariat avec le secteur public",
        "Guyane",
        "Nouvelle-Calédonie",
        "Autre"
    ]

    # Sélection du programme EN DEHORS du formulaire pour permettre le conditionnement dynamique
    st.markdown("### Votre retour d'expérience")

    # Sélection de l'innovation
    innovation_selectionnee = st.selectbox(
        "Innovation concernée *",
        options=[""] + df["Innovation identifiée"].tolist(),
        help="Sélectionnez l'innovation sur laquelle vous souhaitez partager votre expérience",
        key="rex_innovation"
    )

    # Sélection du programme
    col_prog1, col_prog2 = st.columns([1, 1])

    with col_prog1:
        programme_selectionne = st.selectbox(
            "Votre programme *",
            options=PROGRAMMES_WWF,
            help="Sélectionnez le programme auquel vous appartenez",
            key="rex_programme"
        )

    with col_prog2:
        # Champ pour préciser si "Autre" est sélectionné
        programme_autre = st.text_input(
            "Précisez votre programme *" if programme_selectionne == "Autre" else "Précisez votre programme",
            placeholder="Ex: Communication, Finance, etc." if programme_selectionne == "Autre" else "Sélectionnez 'Autre' pour activer ce champ",
            disabled=(programme_selectionne != "Autre"),
            help="Ce champ est actif uniquement si vous sélectionnez 'Autre'",
            key="rex_programme_autre"
        )

    # Formulaire pour le reste des champs
    with st.form("formulaire_retour_experience", clear_on_submit=True):
        # Date optionnelle
        col_date1, col_date2 = st.columns([1, 1])

        with col_date1:
            date_test = st.date_input(
                "Date du test ou de la réflexion (optionnel)",
                value=None,
                help="Indiquez la date approximative de votre expérience avec cette innovation"
            )

        with col_date2:
            st.empty()  # Pour équilibrer la mise en page

        # Zone de texte pour le retour d'expérience
        st.markdown("#### Décrivez votre expérience")
        retour_experience = st.text_area(
            "Votre retour d'expérience *",
            placeholder="Partagez ici votre expérience avec cette innovation :\n"
                        "- Dans quel contexte l'avez-vous testée ou envisagée ?\n"
                        "- Quels ont été les résultats ou observations ?\n"
                        "- Quelles difficultés avez-vous rencontrées ?\n"
                        "- Quels conseils donneriez-vous à d'autres programmes ?",
            height=200,
            help="Soyez aussi précis que possible pour aider vos collègues"
        )

        # Contact optionnel
        st.markdown("#### Vos coordonnées (optionnel)")
        st.caption("Laisser vos coordonnées permet à d'autres collègues de vous contacter pour en savoir plus.")

        col_contact1, col_contact2 = st.columns([1, 1])

        with col_contact1:
            nom_contact = st.text_input(
                "Nom / Prénom",
                placeholder="Ex: Marie Dupont"
            )

        with col_contact2:
            email_contact = st.text_input(
                "Email",
                placeholder="Ex: mdupont@wwf.fr"
            )

        # Bouton de soumission
        submitted_rex = st.form_submit_button("Envoyer mon retour d'expérience", type="primary")

        if submitted_rex:
            # Validation
            erreurs = []
            if not innovation_selectionnee:
                erreurs.append("Veuillez sélectionner une innovation")
            if programme_selectionne == "Autre" and not programme_autre:
                erreurs.append("Veuillez préciser votre programme")
            if not retour_experience or len(retour_experience.strip()) < 10:
                erreurs.append("Veuillez décrire votre expérience (minimum 10 caractères)")

            if erreurs:
                for erreur in erreurs:
                    st.error(erreur)
            else:
                # Sauvegarder le retour d'expérience
                fichier_rex = "retours_experience.csv"
                from datetime import datetime
                import os

                # Déterminer le programme final
                programme_final = programme_autre if programme_selectionne == "Autre" else programme_selectionne

                # Créer le DataFrame avec le nouveau retour
                nouveau_rex = pd.DataFrame({
                    'Date_soumission': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Innovation': [innovation_selectionnee],
                    'Programme': [programme_final],
                    'Date_test': [str(date_test) if date_test else ""],
                    'Retour_experience': [retour_experience],
                    'Nom_contact': [nom_contact if nom_contact else ""],
                    'Email_contact': [email_contact if email_contact else ""]
                })

                # Ajouter au fichier existant ou créer un nouveau fichier
                if os.path.exists(fichier_rex):
                    df_rex = pd.read_csv(fichier_rex)
                    df_rex = pd.concat([df_rex, nouveau_rex], ignore_index=True)
                else:
                    df_rex = nouveau_rex

                df_rex.to_csv(fichier_rex, index=False)
                st.success("Merci pour votre retour d'expérience ! Il a été enregistré avec succès.")
                st.balloons()

    # Section pour consulter les retours existants
    st.markdown("---")
    st.markdown("### Consulter les retours d'expérience")

    fichier_rex = "retours_experience.csv"
    import os

    if os.path.exists(fichier_rex):
        df_rex_existants = pd.read_csv(fichier_rex)

        if not df_rex_existants.empty:
            # Filtre par innovation
            innovations_avec_rex = df_rex_existants['Innovation'].unique().tolist()

            filtre_innovation = st.selectbox(
                "Filtrer par innovation",
                options=["Toutes les innovations"] + sorted(innovations_avec_rex),
                key="rex_filtre"
            )

            if filtre_innovation != "Toutes les innovations":
                df_rex_affiche = df_rex_existants[df_rex_existants['Innovation'] == filtre_innovation]
            else:
                df_rex_affiche = df_rex_existants

            st.caption(f"{len(df_rex_affiche)} retour(s) d'expérience")

            # Afficher les retours sous forme de cartes
            for idx, row in df_rex_affiche.iterrows():
                with st.expander(f"**{row['Innovation']}** - {row['Programme']} ({row['Date_soumission'][:10]})"):
                    if row.get('Date_test') and str(row['Date_test']) != "":
                        st.caption(f"Date du test/réflexion : {row['Date_test']}")

                    st.markdown(row['Retour_experience'])

                    if row.get('Nom_contact') and str(row['Nom_contact']) != "":
                        st.markdown("---")
                        contact_info = f"**Contact** : {row['Nom_contact']}"
                        if row.get('Email_contact') and str(row['Email_contact']) != "":
                            contact_info += f" ({row['Email_contact']})"
                        st.markdown(contact_info)
        else:
            st.info("Aucun retour d'expérience n'a encore été partagé. Soyez le premier !")
    else:
        st.info("Aucun retour d'expérience n'a encore été partagé. Soyez le premier !")

    # Statistiques des retours
    if os.path.exists(fichier_rex):
        df_rex_stats = pd.read_csv(fichier_rex)
        if not df_rex_stats.empty and len(df_rex_stats) >= 3:
            st.markdown("---")
            st.markdown("### Statistiques des retours")

            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.metric("Total retours", len(df_rex_stats))

            with col_stat2:
                nb_innovations = df_rex_stats['Innovation'].nunique()
                st.metric("Innovations concernées", nb_innovations)

            with col_stat3:
                nb_programmes = df_rex_stats['Programme'].nunique()
                st.metric("Programmes contributeurs", nb_programmes)

# --- FOOTER ---
st.sidebar.markdown("---")
if st.sidebar.button("Recharger les données"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("""
<div style="background-color: #e8f4f8; border-left: 4px solid #31708f; padding: 12px 16px; border-radius: 4px; color: #31708f; font-size: 14px;">
    <strong>Astuce</strong> : Vous pouvez modifier le fichier Excel et cliquer sur 'Recharger les données' afin de mettre à jour les visualisations
</div>
""", unsafe_allow_html=True)
