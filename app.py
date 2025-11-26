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
    Cette visualisation presente **{len(df)} innovations** evaluees selon **{len(criteres)} criteres**.
    Les echelles de notation varient selon les criteres.
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Vue d'ensemble", "Portefeuille d'innovations","Comparaison", "Analyse par critère", "Donnees", "Dictionnaire", "Veille"])

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
        df_heatmap_normalized = df_heatmap.copy()
        for col in df_heatmap.columns:
            max_crit = max_par_critere.get(col, df_heatmap[col].replace(-1, np.nan).max())
            if max_crit and max_crit > 0:
                df_heatmap_normalized[col] = df_heatmap[col].apply(
                    lambda x: x / max_crit if x >= 0 else -1
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

        # Colorscale avec couleurs plus contrastees (rouge fonce -> jaune -> vert fonce)
        colorscale = [
            [0, '#d9d9d9'],       # -1 (N/A) = gris
            [0.001, '#d73027'],   # 0% du max = rouge fonce
            [0.25, '#fc8d59'],    # 25% = orange
            [0.5, '#fee08b'],     # 50% = jaune
            [0.75, '#91cf60'],    # 75% = vert clair
            [1, '#1a9850']        # 100% du max = vert fonce
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=df_heatmap_normalized.values,
            x=labels_tronques,
            y=df_heatmap.index,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            text=df_heatmap.values,  # Afficher les valeurs originales
            texttemplate='%{text}',
            textfont={"size": 10, "color": "black"},
            colorbar=dict(
                title="Intensite",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            ),
            customdata=customdata_matrix,
            hovertemplate='<b>%{y}</b><br>%{customdata}<br>Score: %{text}<extra></extra>'
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

        # Calculer le max global pour l'echelle du radar
        max_val_radar = df_filtered[criteres].max().max()

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

        # Creer les tickvals dynamiques
        tick_vals_radar = list(range(int(max_val_radar) + 1))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_val_radar],
                    tickvals=tick_vals_radar
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

    # Statistiques globales avec mini-echelles
    st.subheader("Score moyen par critere")

    with st.expander("Comment lire ces scores ?"):
        st.markdown("""
        **Calcul du score moyen :**
        - Chaque innovation est evaluee sur plusieurs criteres
        - Le score moyen correspond a la **moyenne arithmetique** des scores attribues a toutes les innovations pour ce critere
        - L'echelle de notation varie selon les criteres (ex: 0-2 ou 0-1)

        **Lecture de l'echelle :**
        - Le curseur noir indique la position de la moyenne sur l'echelle du critere
        - Les valeurs min et max affichees correspondent aux bornes reelles du critere
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
                    st.info(col3_val)
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
        st.warning("Le dictionnaire des variables n'a pas pu etre charge.")

# ================== ONGLET 7: VEILLE ==================
with tab7:
    st.subheader("Veille et ressources")

    st.markdown("""
    Cet onglet regroupe les **ressources externes** utiles pour la veille sur l'innovation
    et les donnees environnementales. Cliquez sur les cartes pour acceder aux sources.
    """)

    st.markdown("---")

    # Section des ressources
    st.markdown("### Ressources disponibles")

    # Creer des cartes visuelles pour les ressources
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00693E 0%, #4A7C59 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            height: 280px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top:0; color:white;">Innovation Example Repository - WWF</h3>
            <p style="font-size: 14px; line-height: 1.6;">
                Plateforme Innovation Hub presentant des innovations du reseau WWF.
                Decouvrez des cas concrets d'innovations mises en oeuvre pour la conservation.
            </p>
            <p style="font-size: 12px; opacity: 0.8; margin-bottom: 15px;">
                Source : Impact Hub / WWF
            </p>
            <a href="https://sites.google.com/impacthub.net/innovation-wwf/our-impact/stories-repository?authuser=0"
               target="_blank"
               style="
                   background-color: white;
                   color: #00693E;
                   padding: 10px 20px;
                   border-radius: 25px;
                   text-decoration: none;
                   font-weight: bold;
                   display: inline-block;
               ">
                Acceder a la ressource
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2E7D9A 0%, #7A9D96 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            height: 280px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top:0; color:white;">Catalogue OFB</h3>
            <p style="font-size: 14px; line-height: 1.6;">
                Catalogue de donnees geographiques de l'Office Francais de la Biodiversite (OFB).
                Acces aux donnees environnementales et cartographiques nationales.
            </p>
            <p style="font-size: 12px; opacity: 0.8; margin-bottom: 15px;">
                Source : Office Francais de la Biodiversite
            </p>
            <a href="https://data.ofb.fr/catalogue/Donnees-geographiques-OFB/fre/catalog.search#/home"
               target="_blank"
               style="
                   background-color: white;
                   color: #2E7D9A;
                   padding: 10px 20px;
                   border-radius: 25px;
                   text-decoration: none;
                   font-weight: bold;
                   display: inline-block;
               ">
                Acceder a la ressource
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section pour ajouter des ressources
    st.markdown("### Suggerer une ressource")

    # Formulaire de suggestion avec sauvegarde dans un fichier CSV
    with st.form("formulaire_suggestion", clear_on_submit=True):
        nom_ressource = st.text_input("Nom de la ressource *", placeholder="Ex: Portail de donnees biodiversite")
        url_ressource = st.text_input("URL *", placeholder="https://...")
        description_ressource = st.text_area("Description", placeholder="Decrivez brievement cette ressource...")
        source_ressource = st.text_input("Source / Organisme", placeholder="Ex: Ministere de l'Environnement")

        submitted = st.form_submit_button("Envoyer la suggestion")

        if submitted:
            if nom_ressource and url_ressource:
                # Fichier CSV pour stocker les suggestions
                fichier_suggestions = "suggestions_veille.csv"
                from datetime import datetime
                import os

                # Creer le DataFrame avec la nouvelle suggestion
                nouvelle_suggestion = pd.DataFrame({
                    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Nom': [nom_ressource],
                    'URL': [url_ressource],
                    'Description': [description_ressource],
                    'Source': [source_ressource],
                    'Statut': ['En attente']
                })

                # Ajouter au fichier existant ou creer un nouveau fichier
                if os.path.exists(fichier_suggestions):
                    df_suggestions = pd.read_csv(fichier_suggestions)
                    df_suggestions = pd.concat([df_suggestions, nouvelle_suggestion], ignore_index=True)
                else:
                    df_suggestions = nouvelle_suggestion

                df_suggestions.to_csv(fichier_suggestions, index=False)
                st.success("Merci ! Votre suggestion a été enregistrée et sera examinée par l'administrateur.")
            else:
                st.error("Veuillez remplir au minimum le nom et l'URL de la ressource.")

    # Afficher les suggestions en attente (pour l'administrateur)
    fichier_suggestions = "suggestions_veille.csv"
    import os
    if os.path.exists(fichier_suggestions):
        with st.expander("Voir les suggestions en attente"):
            df_suggestions = pd.read_csv(fichier_suggestions)
            st.dataframe(df_suggestions, use_container_width=True, hide_index=True)
            st.caption(f"{len(df_suggestions)} suggestion(s) enregistree(s)")

    # Resume des ressources
    st.markdown("---")
    st.markdown("### Resume des ressources")

    df_ressources = pd.DataFrame({
        'Ressource': ['Innovation Stories - WWF', 'Catalogue OFB'],
        'Type': ['Innovation', 'Donnees geographiques'],
        'Organisme': ['WWF / Impact Hub', 'Office Francais de la Biodiversite'],
        'Accès': ['Gratuit', 'Gratuit']
    })

    st.dataframe(df_ressources, use_container_width=True, hide_index=True)

# --- FOOTER ---
st.sidebar.markdown("---")
if st.sidebar.button("Recharger les donnees"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("Astuce : Vous pouvez modifier le fichier Excel et cliquer sur 'Recharger les donnees' afin de mettre a jour les visualisations")
