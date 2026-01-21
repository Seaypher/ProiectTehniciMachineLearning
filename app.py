import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import io




# ----------------------------------------------------------------- PROIECTUL INITIAL


st.set_page_config(page_title="Proiect Tehnici ML", page_icon="icon.png", layout="wide")

st.title("Proiect Tehnici de Machine Learning")

# 1.
st.header("1. Incarca un fisier CSV/Excel")
uploaded_file = st.file_uploader("Incarca un fisier CSV sau Excel", type=['csv', 'xlsx'])
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Fisier citit cu succes!")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Eroare la citirea fisierului: {str(e)}")

if df is not None:
    st.subheader("Filtrare Date")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Slidere pentru val numerice
    numeric_filters = {}
    for col in numeric_cols:
        # Eliminam NaN pentru slidere
        col_data = df[col].dropna()
        if col_data.empty:
            st.warning(f"Coloana numerica '{col}' contine doar valori lipsa si va fi ignorata pentru filtrare.")
            continue

        min_val, max_val = float(col_data.min()), float(col_data.max())
        val_range = st.slider(
            f"Filtrare {col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        numeric_filters[col] = val_range

    # Multiselect pentru categoric
    cat_filters = {}
    for col in cat_cols:
        selected = st.multiselect(
            f"Filtrare {col}",
            options=df[col].dropna().unique(),
            default=list(df[col].dropna().unique())
        )
        cat_filters[col] = selected

    # Aplicare filtre
    df_filtered = df.copy()
    for col, (min_val, max_val) in numeric_filters.items():
        df_filtered = df_filtered[(df_filtered[col] >= min_val) & (df_filtered[col] <= max_val)]
    for col, selected in cat_filters.items():
        df_filtered = df_filtered[df_filtered[col].isin(selected)]

    st.markdown(f"**Numar randuri inainte de filtrare:** {df.shape[0]}")
    st.markdown(f"**Numar randuri dupa filtrare:** {df_filtered.shape[0]}")
    st.dataframe(df_filtered, use_container_width=True)

#2.
if df is not None:
    st.header("2. Statistici Descriptive")

    st.markdown(f"**Numar randuri si coloane:** {df.shape}")
    st.markdown("**Tipuri de date per coloana:**")
    st.dataframe(df.dtypes, use_container_width=True)

    st.subheader("Valori Lipsa")
    na_counts = df.isnull().sum()
    na_percent = (na_counts / len(df)) * 100
    na_df = pd.DataFrame({"Missing": na_counts, "Percent": na_percent})
    st.dataframe(na_df[na_df['Missing'] > 0], use_container_width=True)

    # Grafic valori lipsa
    fig_na = px.bar(na_df, x=na_df.index, y="Missing", text="Percent", title="Valori lipsa per coloana")
    fig_na.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_na, use_container_width=True)

    st.subheader("Statistici pentru coloane numerice")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

# 3.
if df is not None and numeric_cols:
    st.header("3. Histograma si Boxplot")

    col_hist = st.selectbox("Selecteaza coloana numerica", numeric_cols)
    n_bins = st.slider("Numar bins histograma", 10, 100, 20)

    # Histogram
    fig_hist = px.histogram(df_filtered, x=col_hist, nbins=n_bins, title=f"Histograma: {col_hist}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Boxplot
    fig_box = px.box(df_filtered, y=col_hist, title=f"Boxplot: {col_hist}", points="all")
    st.plotly_chart(fig_box, use_container_width=True)

    # Statistici
    mean_val = df_filtered[col_hist].mean()
    median_val = df_filtered[col_hist].median()
    std_val = df_filtered[col_hist].std()
    st.markdown(f"**Media:** {mean_val:.2f}, **Mediana:** {median_val:.2f}, **Deviatie standard:** {std_val:.2f}")

# 4.
if df is not None and cat_cols:
    st.header("4. Analiza Coloane Categorice")

    col_cat = st.selectbox("Selecteaza coloana categorica", cat_cols)
    counts = df_filtered[col_cat].value_counts()
    percents = df_filtered[col_cat].value_counts(normalize=True) * 100
    cat_df = pd.DataFrame({"Frecvența": counts, "Procent": percents.round(2)})
    st.dataframe(cat_df, use_container_width=True)

    fig_bar = px.bar(cat_df, x=cat_df.index, y="Frecvența", text="Procent", title=f"Count Plot: {col_cat}")
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

# 5.
if df is not None and numeric_cols:
    st.header("5. Corelatie si Outlieri")

    st.subheader("Matrice Corelatie")
    corr_matrix = df_filtered[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", title="Heatmap Corelatie")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Scatter Plot intre doua variabile")
    col_x = st.selectbox("Variabila X", numeric_cols, index=0)
    col_y = st.selectbox("Variabila Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    fig_scatter = px.scatter(df_filtered, x=col_x, y=col_y, trendline="ols",
                             title=f"Scatter Plot: {col_x} vs {col_y}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    pearson_corr = df_filtered[col_x].corr(df_filtered[col_y])
    st.markdown(f"**Coeficient corelatie Pearson ({col_x}, {col_y}): {pearson_corr:.2f}**")

    st.subheader("Detectie Outlieri cu IQR")
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_filtered[(df_filtered[col] < lower) | (df_filtered[col] > upper)]
        pct_out = len(outliers) / len(df_filtered) * 100
        outlier_summary.append({
            "Coloana": col,
            "Numar outlieri": len(outliers),
            "Procent outlieri": round(pct_out, 2)
        })

        # Vizualizare outlieri
        fig_out = px.box(df_filtered, y=col, title=f"Outlieri: {col}", points="all")
        st.plotly_chart(fig_out, use_container_width=True)

    st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)























































# ------------------------------------------------------------------- CONTINUARE PROIECT


# PARTEA 1 - SETAREA PROBLEMEI

if df is not None:

    st.header("PARTEA 1 - Selectare Date pentru Modelare ML")

    # Selectarea coloanei tinta
    st.subheader("Selectare Coloana Tinta (Target)")

    # Obtinem toate coloanele din dataframe
    all_columns = df.columns.tolist()

    # Selectarea coloanei tinta
    target_column = st.selectbox(
        "Selecteaza coloana tinta (variabila dependenta)",
        all_columns,
        key="target_select"
    )

    # Determinam tipul problemei (clasificare sau regresie)
    if target_column:
        # Verificam daca coloana tinta are suficiente date
        if df[target_column].dropna().empty:
            st.error(f"Coloana tinta '{target_column}' este goala sau contine doar valori lipsa.")
            st.stop()

        # Identificam tipul de date al coloanei tinta
        target_dtype = str(df[target_column].dtype)

        # Verificam daca este problema de clasificare sau regresie
        is_classification = False
        if target_dtype in ['object', 'category', 'bool']:
            is_classification = True
            st.info("Problema identificata: CLASIFICARE (coloana tinta este categoriala)")
        elif target_dtype in ['int64', 'float64']:
            # Verificam daca valorile sunt discrete (putine valori unice) pentru clasificare
            unique_values = df[target_column].nunique()
            # Excludem valorile NaN din calcul
            unique_non_nan = df[target_column].dropna().nunique()

            if unique_non_nan < 10 and unique_non_nan > 1:  # Daca sunt putine valori unice, probabil e clasificare
                is_classification = True
                st.info(f"Problema identificata: CLASIFICARE ({unique_non_nan} clase unice)")
            elif unique_non_nan == 1:
                st.error(
                    f"Coloana tinta '{target_column}' are doar o singura valoare unica. Nu se poate antrena model.")
                st.stop()
            else:
                st.info("Problema identificata: REGRESIE (coloana tinta este numerica continua)")
        else:
            st.error("Tip de date neacceptat pentru coloana tinta. Te rog alege o coloana numerica sau categoriala.")
            st.stop()

    # Selectarea feature-urilor
    st.subheader("Selectare Feature-uri (Variabile Independente)")

    # Excludem coloana tinta din lista de feature-uri
    feature_options = [col for col in all_columns if col != target_column]

    # Validare: verificam ca exista feature-uri disponibile
    if not feature_options:
        st.error("Nu exista alte coloane disponibile pentru a fi folosite ca feature-uri.")
        st.stop()

    # Optiuni pentru selectare feature-uri
    selection_method = st.radio(
        "Metoda de selectare a feature-urilor:",
        ["Selecteaza toate coloanele", "Selecteaza manual", "Exclude coloane"],
        key="feature_selection_method"
    )

    selected_features = []

    if selection_method == "Selecteaza toate coloanele":
        selected_features = feature_options
        st.write(f"Selectate toate {len(selected_features)} feature-uri")

    elif selection_method == "Selecteaza manual":
        selected_features = st.multiselect(
            "Alege feature-urile pentru model:",
            feature_options,
            default=feature_options[:min(5, len(feature_options))] if feature_options else [],
            key="manual_feature_select"
        )

    elif selection_method == "Exclude coloane":
        exclude_features = st.multiselect(
            "Alege coloanele de exclus:",
            feature_options,
            key="exclude_feature_select"
        )
        selected_features = [col for col in feature_options if col not in exclude_features]

    # Validare: verificam ca am selectat cel putin un feature
    if not selected_features:
        st.error("Trebuie sa selectezi cel putin un feature pentru antrenarea modelului.")
        st.stop()

    # Validare: verificam ca feature-urile nu sunt goale si au variatie
    invalid_features = []
    for col in selected_features:
        # Verificam daca coloana este goala
        if df[col].dropna().empty:
            invalid_features.append(f"{col} (goala)")
        # Verificam daca coloana numerica are variatie (nu toate valorile sunt aceleasi)
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().nunique()
            if unique_vals <= 1:
                invalid_features.append(f"{col} (fara variatie, toate valorile sunt aceleasi)")

    if invalid_features:
        st.error(f"Urmatoarele coloane au probleme: {', '.join(invalid_features)}")
        st.write("Te rog elimina aceste coloane din selectie sau verifica datele.")
        st.stop()

    # Afisam datele selectate
    st.subheader("Rezumat Setare Problema")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Coloana Tinta:**", target_column)
        st.write("**Tip Problema:**", "Clasificare" if is_classification else "Regresie")
        if is_classification:
            # Folosim doar valori non-NaN pentru calcul
            unique_classes = df[target_column].dropna().nunique()
            st.write(f"**Numar Clase:** {unique_classes}")
            if unique_classes > 0:
                st.write("**Distributie Clase:**")
                class_dist = df[target_column].value_counts(dropna=True)
                st.write(class_dist)

    with col2:
        st.write(f"**Numar Feature-uri:** {len(selected_features)}")
        st.write("**Feature-uri Selectate:**")
        # Afisam doar primele 10 feature-uri pentru a nu ingreuna interfata
        display_features = selected_features[:10]
        for feat in display_features:
            # Verificam tipul fiecarui feature
            feat_dtype = str(df[feat].dtype)
            unique_vals = df[feat].dropna().nunique()
            st.write(f"- {feat} ({feat_dtype}, {unique_vals} valori unice)")

        if len(selected_features) > 10:
            st.write(f"...si inca {len(selected_features) - 10} altele")

    # Pregatim datele pentru urmatoarele etape
    X = df[selected_features].copy()  # Feature-uri (facem copie pentru siguranta)
    y = df[target_column].copy()  # Target (facem copie pentru siguranta)

    # Salvam variabilele in session state pentru a le folosi in urmatoarele parti
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['is_classification'] = is_classification
    st.session_state['target_column'] = target_column
    st.session_state['selected_features'] = selected_features

# PARTEA 2 - PREPROCESARE CU PIPELINE

if 'X' in st.session_state and 'y' in st.session_state:

    st.header("PARTEA 2 - Preprocesare Date")

    X = st.session_state['X']
    y = st.session_state['y']
    is_classification = st.session_state['is_classification']

    # Identificam tipurile de coloane
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Afisam tipurile de coloane gasite
    st.subheader("Tipuri de Coloane Identificate")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Coloane Numerice:**")
        if numeric_features:
            for feat in numeric_features:
                # Verificam variatia in fiecare coloana numerica
                unique_vals = X[feat].dropna().nunique()
                min_val = X[feat].min() if not X[feat].empty else 0
                max_val = X[feat].max() if not X[feat].empty else 0
                st.write(f"- {feat}: {unique_vals} valori, range: [{min_val:.2f}, {max_val:.2f}]")
        else:
            st.write("Niciuna")

    with col2:
        st.write("**Coloane Categorice:**")
        if categorical_features:
            for feat in categorical_features:
                unique_vals = X[feat].dropna().nunique()
                st.write(f"- {feat}: {unique_vals} categorii unice")
        else:
            st.write("Niciuna")

    # Setari pentru preprocesare
    st.subheader("Setari Preprocesare")

    # Configurari pentru coloane numerice
    if numeric_features:
        st.markdown("#### Pentru Coloanele Numerice:")

        # Selectare metoda de imputare pentru valori lipsa
        numeric_imputer = st.selectbox(
            "Metoda de imputare pentru valori lipsa (numerice):",
            ["mean", "median", "most_frequent"],
            key="numeric_imputer"
        )

        # Selectare metoda de scalare
        numeric_scaler = st.selectbox(
            "Metoda de scalare (numerice):",
            ["StandardScaler", "MinMaxScaler", "Fara scalare"],
            key="numeric_scaler"
        )

    # Configurari pentru coloane categorice
    if categorical_features:
        st.markdown("#### Pentru Coloanele Categorice:")

        # Selectare metoda de imputare pentru valori lipsa
        categorical_imputer = st.selectbox(
            "Metoda de inlocuire pentru valori lipsa (categorice):",
            ["most_frequent", "constant_0"],
            key="categorical_imputer"
        )

        # Selectare metoda de encoding
        categorical_encoder = st.selectbox(
            "Metoda de encoding (categorice):",
            ["OneHotEncoder", "OrdinalEncoder"],
            key="categorical_encoder"
        )

    # Optiuni avansate pentru preprocesare
    st.subheader("Optiuni Avansate de Preprocesare")

    # Optiune pentru eliminare outlieri - folosim o cheie unica
    remove_outliers = st.checkbox("Elimina outlieri (folosind IQR method)", value=False, key="remove_outliers_checkbox")

    # Salvam valoarea in session state fara a modifica widget-ul
    if 'remove_outliers' not in st.session_state:
        st.session_state['remove_outliers'] = remove_outliers
    else:
        # Actualizam doar daca valoarea s-a schimbat
        st.session_state['remove_outliers'] = remove_outliers

    # Optiune pentru selectie feature-uri - folosim o cheie unica
    feature_selection = st.checkbox("Aplica selectie automata de feature-uri", value=False,
                                    key="feature_selection_checkbox")

    # Salvam valoarea in session state fara a modifica widget-ul
    if 'feature_selection' not in st.session_state:
        st.session_state['feature_selection'] = feature_selection
    else:
        # Actualizam doar daca valoarea s-a schimbat
        st.session_state['feature_selection'] = feature_selection

    if feature_selection:
        # Limitam numarul maxim de feature-uri la numarul real disponibil
        max_features = len(st.session_state['selected_features'])
        if max_features < 2:
            st.warning("Nu poti aplica selectie de feature-uri cu mai putin de 2 feature-uri disponibile.")
            st.session_state['feature_selection'] = False
        else:
            k_features = st.slider(
                "Numar de feature-uri de selectat (k):",
                min_value=1,
                max_value=max_features,
                value=min(5, max_features),
                key="k_features_slider"
            )
            # Salvam valoarea in session state
            st.session_state['k_features'] = k_features

    # Creare transformers pentru pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

    # Definim transformers pentru coloane numerice
    numeric_transformer_steps = []

    # Adaugam imputer pentru valori lipsa doar daca exista coloane numerice
    if numeric_features:
        if numeric_imputer == "most_frequent":
            numeric_imputer_strategy = "most_frequent"
        else:
            numeric_imputer_strategy = numeric_imputer

        numeric_transformer_steps.append(('imputer', SimpleImputer(strategy=numeric_imputer_strategy)))

        # Adaugam scaler daca este selectat
        if numeric_scaler == "StandardScaler":
            numeric_transformer_steps.append(('scaler', StandardScaler()))
        elif numeric_scaler == "MinMaxScaler":
            numeric_transformer_steps.append(('scaler', MinMaxScaler()))
        # Daca este "Fara scalare", nu adaugam niciun scaler

    # Definim transformers pentru coloane categorice
    categorical_transformer_steps = []

    if categorical_features:
        # Adaugam imputer pentru valori lipsa
        if categorical_imputer == "constant_0":
            categorical_transformer_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
        else:  # most_frequent
            categorical_transformer_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))

        # Adaugam encoder
        if categorical_encoder == "OneHotEncoder":
            categorical_transformer_steps.append(
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        else:  # OrdinalEncoder
            categorical_transformer_steps.append(
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))

    # Creare ColumnTransformer
    transformers = []

    if numeric_features and numeric_transformer_steps:
        numeric_pipeline = Pipeline(steps=numeric_transformer_steps)
        transformers.append(('num', numeric_pipeline, numeric_features))

    if categorical_features and categorical_transformer_steps:
        categorical_pipeline = Pipeline(steps=categorical_transformer_steps)
        transformers.append(('cat', categorical_pipeline, categorical_features))

    # Daca nu avem transformers pentru niciun tip de date, folosim un passthrough
    if not transformers:
        st.warning("Nu s-au creat transformers pentru preprocesare. Se va folosi datele in forma originala.")
        preprocessor = ColumnTransformer(transformers=[('passthrough', 'passthrough', selected_features)])
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Salvam preprocessor-ul in session state
    st.session_state['preprocessor'] = preprocessor
    st.session_state['numeric_features'] = numeric_features
    st.session_state['categorical_features'] = categorical_features

    st.success("Configuratia de preprocesare a fost salvata!")

# PARTEA 3 - SPLIT DATE

if 'X' in st.session_state and 'y' in st.session_state:

    st.header("PARTEA 3 - Impartire Date (Train/Test Split)")

    X = st.session_state['X']
    y = st.session_state['y']

    # Validare: verificam ca avem suficiente date pentru split
    min_samples_required = 10  # Minim 10 mostre pentru a putea face split
    if len(X) < min_samples_required:
        st.error(
            f"Numarul de mostre ({len(X)}) este prea mic pentru a imparti datele. Minim {min_samples_required} mostre sunt necesare.")
        st.stop()

    # Configurare split
    st.subheader("Configurare Split Date")

    # Selectare tip split - folosim o cheie unica pentru widget
    split_type_radio = st.radio(
        "Tipul de impartire a datelor:",
        ["Train/Test (80/20)", "Train/Validation/Test (70/15/15)"],
        key="split_type_radio"
    )

    # Setare random state - folosim o cheie unica pentru widget
    random_state_input = st.number_input(
        "Random State (pentru reproducibilitate):",
        min_value=0,
        max_value=1000,
        value=42,
        key="random_state_input"
    )

    # Buton pentru a realiza split-ul
    if st.button("Realizeaza Impartirea Datelor", key="split_button"):

        from sklearn.model_selection import train_test_split

        # Verificam daca avem suficiente date pentru stratificare (pentru clasificare)
        can_stratify = False
        if st.session_state['is_classification']:
            # Pentru stratificare, fiecare clasa trebuie sa aiba cel putin 2 mostre
            class_counts = y.value_counts()
            if all(count >= 2 for count in class_counts):
                can_stratify = True
            else:
                st.warning("Unele clase au mai putin de 2 mostre. Stratificarea nu este posibila.")

        try:
            # Folosim variabilele temporare din widget-uri, nu le salvam direct in session state
            split_type_value = split_type_radio
            random_state_value = random_state_input

            if split_type_value == "Train/Test (80/20)":
                # Split 80/20
                if can_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=random_state_value, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=random_state_value
                    )

                # Validare: verificam ca seturile nu sunt goale
                if len(X_train) == 0 or len(X_test) == 0:
                    st.error("Setul de train sau test este gol. Verifica datele si incearca din nou.")
                    st.stop()

                # Salvam in session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['split_type_value'] = "80/20"  # Salvam ca string, nu ca widget
                st.session_state['random_state_value'] = random_state_value  # Salvam valoarea, nu widget-ul

                st.success(
                    f"Datele au fost impartite: Train={len(X_train)} ({len(X_train) / len(X) * 100:.1f}%), Test={len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

                # Afisam distributia pentru clasificare
                if st.session_state['is_classification']:
                    st.subheader("Distributia Claselor in Seturi")

                    try:
                        train_dist = y_train.value_counts(normalize=True) * 100
                        test_dist = y_test.value_counts(normalize=True) * 100

                        dist_df = pd.DataFrame({
                            'Train %': train_dist,
                            'Test %': test_dist
                        }).round(2)

                        st.dataframe(dist_df, use_container_width=True)
                    except:
                        st.write("Nu se poate calcula distributia claselor.")

            else:  # Train/Validation/Test
                # Verificam daca avem suficiente date pentru 3 seturi
                if len(X) < 30:
                    st.error(f"Numarul de mostre ({len(X)}) este prea mic pentru 3 seturi. Alege Train/Test (80/20).")
                    st.stop()

                # Mai intai split 85/15 pentru a separa test
                if can_stratify:
                    X_train_val, X_test, y_train_val, y_test = train_test_split(
                        X, y, test_size=0.15, random_state=random_state_value, stratify=y
                    )
                else:
                    X_train_val, X_test, y_train_val, y_test = train_test_split(
                        X, y, test_size=0.15, random_state=random_state_value
                    )

                # Apoi split 70/15 din restul pentru train/validation
                if can_stratify:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, test_size=0.176, random_state=random_state_value,  # 0.176 = 15/85
                        stratify=y_train_val
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, test_size=0.176, random_state=random_state_value
                    )

                # Validare: verificam ca seturile nu sunt goale
                if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                    st.error("Unul dintre seturi este gol. Verifica datele si incearca din nou.")
                    st.stop()

                # Salvam in session state
                st.session_state['X_train'] = X_train
                st.session_state['X_val'] = X_val
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_val'] = y_val
                st.session_state['y_test'] = y_test
                st.session_state['split_type_value'] = "70/15/15"  # Salvam ca string, nu ca widget
                st.session_state['random_state_value'] = random_state_value  # Salvam valoarea, nu widget-ul

                train_pct = len(X_train) / len(X) * 100
                val_pct = len(X_val) / len(X) * 100
                test_pct = len(X_test) / len(X) * 100

                st.success(
                    f"Datele au fost impartite: Train={len(X_train)} ({train_pct:.1f}%), Val={len(X_val)} ({val_pct:.1f}%), Test={len(X_test)} ({test_pct:.1f}%)")

                # Afisam distributia pentru clasificare
                if st.session_state['is_classification']:
                    st.subheader("Distributia Claselor in Seturi")

                    try:
                        train_dist = y_train.value_counts(normalize=True) * 100
                        val_dist = y_val.value_counts(normalize=True) * 100
                        test_dist = y_test.value_counts(normalize=True) * 100

                        dist_df = pd.DataFrame({
                            'Train %': train_dist,
                            'Validation %': val_dist,
                            'Test %': test_dist
                        }).round(2)

                        st.dataframe(dist_df, use_container_width=True)
                    except:
                        st.write("Nu se poate calcula distributia claselor.")

        except Exception as e:
            st.error(f"Eroare la impartirea datelor: {str(e)}")
            st.write("Verifica daca datele sunt valide si incearca din nou.")

# PARTEA 4 - SELECTARE SI ANTRENARE MODELE ML

if 'X_train' in st.session_state and 'y_train' in st.session_state:

    st.header("PARTEA 4 - Selectare si Antrenare Modele ML")

    is_classification = st.session_state['is_classification']

    # Selectare algoritmi in functie de tipul problemei
    st.subheader("Selectare Algoritmi de Machine Learning")

    if is_classification:
        # Optiuni pentru clasificare
        model_options = {
            "Logistic Regression": "lr",
            "Random Forest Classifier": "rf",
            "Support Vector Machine (SVM)": "svm",
            "K-Nearest Neighbors (KNN)": "knn"
        }
    else:
        # Optiuni pentru regresie
        model_options = {
            "Linear Regression": "lr",
            "Random Forest Regressor": "rf",
            "Ridge Regression": "ridge",
            "Lasso Regression": "lasso"
        }

    # Selectare modele - folosim o cheie unica pentru widget
    selected_models_multiselect = st.multiselect(
        "Selecteaza algoritmii pentru antrenare (minim 2):",
        list(model_options.keys()),
        default=list(model_options.keys())[:2] if len(model_options) >= 2 else list(model_options.keys()),
        key="model_selection_multiselect"
    )

    # Validare: verificam ca sunt selectate minim 2 modele
    if len(selected_models_multiselect) < 2:
        st.error("Trebuie sa selectezi minim 2 algoritmi pentru comparatie.")
        st.stop()

    # Afisam parametrii pentru fiecare model selectat
    st.subheader("Configurare Hiperparametri")

    model_params = {}

    for model_name in selected_models_multiselect:
        st.markdown(f"**{model_name}**")

        model_key = model_options[model_name]

        if is_classification:
            # Parametri pentru modelele de clasificare
            if model_name == "Logistic Regression":
                col1, col2 = st.columns(2)
                with col1:
                    C = st.number_input("C (inverse regularization strength)",
                                        min_value=0.01, max_value=10.0, value=1.0,
                                        key=f"lr_C_{model_key}")
                with col2:
                    max_iter = st.number_input("Numar maxim iteratii",
                                               min_value=100, max_value=1000, value=100,
                                               key=f"lr_iter_{model_key}")
                model_params[model_key] = {'C': C, 'max_iter': max_iter}

            elif model_name == "Random Forest Classifier":
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.number_input("Numar arbori",
                                                   min_value=10, max_value=500, value=100,
                                                   key=f"rf_n_{model_key}")
                with col2:
                    max_depth = st.number_input("Adancime maxima",
                                                min_value=1, max_value=50, value=10,
                                                key=f"rf_depth_{model_key}")
                with col3:
                    min_samples_split = st.number_input("Min samples split",
                                                        min_value=2, max_value=20, value=2,
                                                        key=f"rf_split_{model_key}")
                model_params[model_key] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split
                }

            elif model_name == "Support Vector Machine (SVM)":
                col1, col2 = st.columns(2)
                with col1:
                    C = st.number_input("C (regularization)",
                                        min_value=0.01, max_value=10.0, value=1.0,
                                        key=f"svm_C_{model_key}")
                with col2:
                    kernel = st.selectbox("Kernel",
                                          ['linear', 'rbf', 'poly'],
                                          key=f"svm_kernel_{model_key}")
                model_params[model_key] = {'C': C, 'kernel': kernel}

            elif model_name == "K-Nearest Neighbors (KNN)":
                col1, col2 = st.columns(2)
                with col1:
                    n_neighbors = st.number_input("Numar vecini (k)",
                                                  min_value=1, max_value=50, value=5,
                                                  key=f"knn_n_{model_key}")
                with col2:
                    weights = st.selectbox("Greutate",
                                           ['uniform', 'distance'],
                                           key=f"knn_weights_{model_key}")
                model_params[model_key] = {'n_neighbors': n_neighbors, 'weights': weights}

        else:
            # Parametri pentru modelele de regresie
            if model_name == "Linear Regression":
                # Linear Regression are putini parametrii
                model_params[model_key] = {}

            elif model_name == "Random Forest Regressor":
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.number_input("Numar arbori",
                                                   min_value=10, max_value=500, value=100,
                                                   key=f"rf_n_{model_key}")
                with col2:
                    max_depth = st.number_input("Adancime maxima",
                                                min_value=1, max_value=50, value=10,
                                                key=f"rf_depth_{model_key}")
                with col3:
                    min_samples_split = st.number_input("Min samples split",
                                                        min_value=2, max_value=20, value=2,
                                                        key=f"rf_split_{model_key}")
                model_params[model_key] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split
                }

            elif model_name == "Ridge Regression":
                col1, col2 = st.columns(2)
                with col1:
                    alpha = st.number_input("Alpha (regularization)",
                                            min_value=0.01, max_value=10.0, value=1.0,
                                            key=f"ridge_alpha_{model_key}")
                with col2:
                    max_iter = st.number_input("Numar maxim iteratii",
                                               min_value=100, max_value=1000, value=100,
                                               key=f"ridge_iter_{model_key}")
                model_params[model_key] = {'alpha': alpha, 'max_iter': max_iter}

            elif model_name == "Lasso Regression":
                col1, col2 = st.columns(2)
                with col1:
                    alpha = st.number_input("Alpha (regularization)",
                                            min_value=0.01, max_value=10.0, value=1.0,
                                            key=f"lasso_alpha_{model_key}")
                with col2:
                    max_iter = st.number_input("Numar maxim iteratii",
                                               min_value=100, max_value=1000, value=100,
                                               key=f"lasso_iter_{model_key}")
                model_params[model_key] = {'alpha': alpha, 'max_iter': max_iter}

    # Buton pentru antrenarea modelelor
    if st.button("Antreneaza Modelele", key="train_models"):

        with st.spinner("Se antreneaza modelele... Aceasta poate dura cateva momente."):

            # Importam bibliotecile necesare
            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.metrics import confusion_matrix, classification_report
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression

            # Definim dictionarul de modele
            models = {}
            pipelines = {}
            results = {}

            # Obtinem datele din session state
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            preprocessor = st.session_state['preprocessor']

            # Preluam optiunile de preprocesare din session state
            remove_outliers = st.session_state.get('remove_outliers', False)
            feature_selection = st.session_state.get('feature_selection', False)
            k_features = st.session_state.get('k_features', 10)

            # Folosim variabila temporara din widget, nu din session state
            selected_models = selected_models_multiselect

            # Cream pipeline-ul final cu preprocesare si model
            for model_name in selected_models:
                model_key = model_options[model_name]

                # Cream modelul cu parametrii specificati
                if is_classification:
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(
                            C=model_params[model_key]['C'],
                            max_iter=model_params[model_key]['max_iter'],
                            random_state=st.session_state.get('random_state_value', 42)
                        )
                    elif model_name == "Random Forest Classifier":
                        model = RandomForestClassifier(
                            n_estimators=model_params[model_key]['n_estimators'],
                            max_depth=model_params[model_key]['max_depth'],
                            min_samples_split=model_params[model_key]['min_samples_split'],
                            random_state=st.session_state.get('random_state_value', 42)
                        )
                    elif model_name == "Support Vector Machine (SVM)":
                        model = SVC(
                            C=model_params[model_key]['C'],
                            kernel=model_params[model_key]['kernel'],
                            random_state=st.session_state.get('random_state_value', 42)
                        )
                    elif model_name == "K-Nearest Neighbors (KNN)":
                        model = KNeighborsClassifier(
                            n_neighbors=model_params[model_key]['n_neighbors'],
                            weights=model_params[model_key]['weights']
                        )
                else:
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    elif model_name == "Random Forest Regressor":
                        model = RandomForestRegressor(
                            n_estimators=model_params[model_key]['n_estimators'],
                            max_depth=model_params[model_key]['max_depth'],
                            min_samples_split=model_params[model_key]['min_samples_split'],
                            random_state=st.session_state.get('random_state_value', 42)
                        )
                    elif model_name == "Ridge Regression":
                        model = Ridge(
                            alpha=model_params[model_key]['alpha'],
                            max_iter=model_params[model_key]['max_iter'],
                            random_state=st.session_state.get('random_state_value', 42)
                        )
                    elif model_name == "Lasso Regression":
                        model = Lasso(
                            alpha=model_params[model_key]['alpha'],
                            max_iter=model_params[model_key]['max_iter'],
                            random_state=st.session_state.get('random_state_value', 42)
                        )

                # Cream pipeline-ul final
                pipeline_steps = [('preprocessor', preprocessor)]

                # Adaugam selectie de feature-uri daca este activata
                if feature_selection:
                    if is_classification:
                        selector = SelectKBest(score_func=f_classif, k=k_features)
                    else:
                        selector = SelectKBest(score_func=f_regression, k=k_features)
                    pipeline_steps.append(('feature_selection', selector))

                # Adaugam modelul
                pipeline_steps.append(('model', model))

                # Cream pipeline-ul
                pipeline = Pipeline(steps=pipeline_steps)

                # Antrenam modelul
                pipeline.fit(X_train, y_train)

                # Realizam predictii
                y_pred = pipeline.predict(X_test)

                # Salvam rezultatele
                models[model_name] = model
                pipelines[model_name] = pipeline

                # Calculam metricile
                if is_classification:
                    # Pentru clasificare
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    results[model_name] = {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'y_pred': y_pred,
                        'y_test': y_test
                    }
                else:
                    # Pentru regresie
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    results[model_name] = {
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'R²': r2,
                        'y_pred': y_pred,
                        'y_test': y_test
                    }

            # Salvam rezultatele in session state
            st.session_state['models'] = models
            st.session_state['pipelines'] = pipelines
            st.session_state['results'] = results
            st.session_state['selected_models_value'] = selected_models  # Salvam valoarea, nu widget-ul

            st.success("Modelele au fost antrenate cu succes!")

# PARTEA 5 - EVALUARE SI COMPARATIE MODELE

# Importam bibliotecile pentru plotare si metrici
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score

if 'results' in st.session_state:

    st.header("PARTEA 5 - Evaluare si Comparatie Modele")

    results = st.session_state['results']
    is_classification = st.session_state['is_classification']

    # Afisam rezultatele pentru fiecare model
    st.subheader("Rezultate Evaluare pe Setul de Test")

    # Cream un dataframe cu toate metricile
    if is_classification:
        # Pentru clasificare
        metrics_df = pd.DataFrame({
            model_name: {
                'Accuracy': res['Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1 Score': res['F1 Score']
            }
            for model_name, res in results.items()
        }).T

        # Adaugam coloana pentru cel mai bun model pentru fiecare metrica
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            best_idx = metrics_df[metric].idxmax()
            metrics_df[f'Best {metric}'] = metrics_df.index == best_idx

    else:
        # Pentru regresie
        metrics_df = pd.DataFrame({
            model_name: {
                'MAE': res['MAE'],
                'MSE': res['MSE'],
                'RMSE': res['RMSE'],
                'R²': res['R²']
            }
            for model_name, res in results.items()
        }).T

        # Adaugam coloana pentru cel mai bun model pentru fiecare metrica
        # Pentru MAE, MSE, RMSE - valorile mai mici sunt mai bune
        # Pentru R² - valorile mai mari sunt mai bune

        for metric in ['MAE', 'MSE', 'RMSE']:
            best_idx = metrics_df[metric].idxmin()
            metrics_df[f'Best {metric}'] = metrics_df.index == best_idx

        best_idx_r2 = metrics_df['R²'].idxmax()
        metrics_df['Best R²'] = metrics_df.index == best_idx_r2

    # Afisam tabelul cu metrici
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    # Selectare metrica pentru determinarea celui mai bun model
    st.subheader("Selectare Metrica pentru Comparatie")

    if is_classification:
        primary_metric = st.selectbox(
            "Selecteaza metrica principala pentru comparatie:",
            ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            key="primary_metric"
        )
    else:
        primary_metric = st.selectbox(
            "Selecteaza metrica principala pentru comparatie:",
            ['R²', 'MAE', 'RMSE'],
            key="primary_metric"
        )

    # Determinam cel mai bun model bazat pe metrica selectata
    if primary_metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R²']:
        # Pentru aceste metrici, valorile mai mari sunt mai bune
        best_model_name = max(results.items(), key=lambda x: x[1][primary_metric])[0]
        best_score = results[best_model_name][primary_metric]
    else:
        # Pentru MAE, MSE, RMSE, valorile mai mici sunt mai bune
        best_model_name = min(results.items(), key=lambda x: x[1][primary_metric])[0]
        best_score = results[best_model_name][primary_metric]

    # Afisam cel mai bun model
    st.success(f"**Cel mai bun model ({primary_metric}): {best_model_name} cu scorul {best_score:.4f}**")

    # Vizualizari pentru fiecare model
    st.subheader("Vizualizari Detaliate per Model")

    # Selectare model pentru vizualizare detaliata
    selected_model_viz = st.selectbox(
        "Selecteaza un model pentru vizualizari detaliate:",
        list(results.keys()),
        key="model_viz_select"
    )

    # Obtinem datele pentru modelul selectat
    model_results = results[selected_model_viz]

    if is_classification:
        # Vizualizari pentru clasificare

        # Matrice de confuzie
        st.markdown(f"#### Matrice de Confuzie - {selected_model_viz}")

        # Calculam matricea de confuzie
        cm = confusion_matrix(model_results['y_test'], model_results['y_pred'])

        # Cream vizualizarea
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predictii')
        ax.set_ylabel('Valori Reale')
        ax.set_title(f'Matrice de Confuzie - {selected_model_viz}')
        st.pyplot(fig)

        # Inchidem figura pentru a evita warning-uri
        plt.close(fig)

        # Raport de clasificare
        st.markdown(f"#### Raport de Clasificare - {selected_model_viz}")

        # Calculam raportul de clasificare
        try:
            class_report = classification_report(model_results['y_test'], model_results['y_pred'], output_dict=True)
            class_report_df = pd.DataFrame(class_report).transpose()

            st.dataframe(class_report_df.style.format("{:.4f}"), use_container_width=True)
        except Exception as e:
            st.warning(f"Nu se poate genera raportul de clasificare: {str(e)}")

    else:
        # Vizualizari pentru regresie

        # Plot predictii vs valori reale
        st.markdown(f"#### Predictii vs Valori Reale - {selected_model_viz}")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Cream scatter plot pentru predictii vs valori reale
        ax.scatter(model_results['y_test'], model_results['y_pred'], alpha=0.5)

        # Adaugam linia de perfecta predictie
        min_val = min(model_results['y_test'].min(), model_results['y_pred'].min())
        max_val = max(model_results['y_test'].max(), model_results['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predictie Perfecta')

        ax.set_xlabel('Valori Reale')
        ax.set_ylabel('Predictii')
        ax.set_title(f'Predictii vs Valori Reale - {selected_model_viz}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Inchidem figura pentru a evita warning-uri
        plt.close(fig)

        # Histograma erorilor
        st.markdown(f"#### Distributia Erorilor - {selected_model_viz}")

        # Calculam erorile
        errors = model_results['y_test'] - model_results['y_pred']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        ax.set_xlabel('Eroare (Real - Predictie)')
        ax.set_ylabel('Frecventa')
        ax.set_title(f'Distributia Erorilor - {selected_model_viz}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Inchidem figura pentru a evita warning-uri
        plt.close(fig)

    # Comparatie vizuala intre modele
    st.subheader("Comparatie Vizuala Intre Modele")

    if is_classification:
        # Pentru clasificare - barchart cu metrici
        fig, ax = plt.subplots(figsize=(10, 6))

        # Pregatim datele pentru plot
        model_names = list(results.keys())
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # Cream un dataframe pentru plot
        plot_data = []
        for model in model_names:
            for metric in metrics_to_plot:
                plot_data.append({
                    'Model': model,
                    'Metrica': metric,
                    'Scor': results[model][metric]
                })

        plot_df = pd.DataFrame(plot_data)

        # Cream grouped bar chart
        x = np.arange(len(model_names))
        width = 0.2

        for i, metric in enumerate(metrics_to_plot):
            metric_scores = [results[model][metric] for model in model_names]
            ax.bar(x + i * width, metric_scores, width, label=metric)

        ax.set_xlabel('Model')
        ax.set_ylabel('Scor')
        ax.set_title('Comparatie Metrice pe Modele')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Inchidem figura pentru a evita warning-uri
        plt.close(fig)

    else:
        # Pentru regresie - barchart cu metrici
        fig, ax = plt.subplots(figsize=(10, 6))

        # Pregatim datele pentru plot
        model_names = list(results.keys())
        metrics_to_plot = ['R²', 'MAE', 'RMSE']

        # Cream un dataframe pentru plot
        plot_data = []
        for model in model_names:
            for metric in metrics_to_plot:
                plot_data.append({
                    'Model': model,
                    'Metrica': metric,
                    'Scor': results[model][metric]
                })

        plot_df = pd.DataFrame(plot_data)

        # Cream grouped bar chart
        x = np.arange(len(model_names))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            metric_scores = [results[model][metric] for model in model_names]
            ax.bar(x + i * width, metric_scores, width, label=metric)

        ax.set_xlabel('Model')
        ax.set_ylabel('Scor')
        ax.set_title('Comparatie Metrice pe Modele')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Inchidem figura pentru a evita warning-uri
        plt.close(fig)

    # Sectiune pentru predictii pe date noi
    st.subheader("Predictii pe Date Noi")

    # Verificam daca avem pipelines salvate
    if 'pipelines' in st.session_state:
        pipelines = st.session_state['pipelines']

        # Creare dataframe pentru predictii
        st.write("Introdu valorile pentru feature-uri pentru a face predictii:")

        # Cream un formular pentru introducere date
        input_data = {}

        # Obtinem feature-urile originale
        selected_features = st.session_state.get('selected_features', [])

        # Cream cate un input pentru fiecare feature
        cols = st.columns(3)  # Impartim in 3 coloane

        for i, feature in enumerate(selected_features):
            col_idx = i % 3
            with cols[col_idx]:
                # Identificam tipul de date al feature-ului
                feature_dtype = str(st.session_state['X'][feature].dtype)

                if feature_dtype in ['int64', 'float64']:
                    # Pentru feature-uri numerice
                    min_val = float(st.session_state['X'][feature].min())
                    max_val = float(st.session_state['X'][feature].max())
                    mean_val = float(st.session_state['X'][feature].mean())

                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    # Pentru feature-uri categorice
                    unique_vals = st.session_state['X'][feature].dropna().unique().tolist()
                    if len(unique_vals) > 10:
                        # Daca sunt prea multe valori, folosim text input
                        input_data[feature] = st.text_input(
                            f"{feature}",
                            value=str(unique_vals[0]) if unique_vals else "",
                            key=f"input_{feature}"
                        )
                    else:
                        # Pentru putine valori, folosim selectbox
                        input_data[feature] = st.selectbox(
                            f"{feature}",
                            options=unique_vals,
                            key=f"input_{feature}"
                        )

        # Buton pentru predictie
        if st.button("Fa Predictii", key="make_predictions"):
            # Cream un dataframe cu datele de input
            input_df = pd.DataFrame([input_data])

            # Facem predictii cu fiecare model
            predictions = {}

            for model_name, pipeline in pipelines.items():
                try:
                    # Realizam predictia
                    if is_classification:
                        pred = pipeline.predict(input_df)[0]

                        # Incercam sa obtinem probabilitatile daca modelul le suporta
                        try:
                            pred_prob = pipeline.predict_proba(input_df)[0]
                            predictions[model_name] = {
                                'Predictie': pred,
                                'Probabilitati': pred_prob.tolist()  # Convertim array-ul la lista
                            }
                        except:
                            predictions[model_name] = {
                                'Predictie': pred,
                                'Probabilitati': []  # Lista goala daca nu exista probabilitati
                            }
                    else:
                        pred = pipeline.predict(input_df)[0]
                        predictions[model_name] = {
                            'Predictie': pred
                        }

                except Exception as e:
                    st.error(f"Eroare la predictia cu {model_name}: {str(e)}")
                    predictions[model_name] = {'Predictie': 'Eroare', 'Probabilitati': []}

            # Afisam predictiile
            st.subheader("Rezultate Predictii")

            if is_classification:
                # Pentru clasificare
                pred_df = pd.DataFrame({
                    'Model': predictions.keys(),
                    'Predictie': [p['Predictie'] for p in predictions.values()],
                    # CORECTIE: Folosim len() pentru a verifica daca lista are elemente
                    'Probabilitate Maxima': [max(p['Probabilitati']) if len(p['Probabilitati']) > 0 else 'N/A' for p in
                                             predictions.values()]
                })
            else:
                # Pentru regresie
                pred_df = pd.DataFrame({
                    'Model': predictions.keys(),
                    'Predictie': [p['Predictie'] for p in predictions.values()]
                })

            st.dataframe(pred_df, use_container_width=True)

            # Afisam predictia modelului cel mai bun
            best_pipeline = pipelines.get(best_model_name)

            if best_pipeline:
                if is_classification:
                    best_pred = best_pipeline.predict(input_df)[0]

                    st.success(f"**Modelul cel mai bun ({best_model_name}) prezice: {best_pred}**")

                    # Incercam sa afisam probabilitatile pentru clasificare
                    try:
                        best_proba = best_pipeline.predict_proba(input_df)[0]

                        if hasattr(best_pipeline, 'classes_'):
                            classes = best_pipeline.classes_
                            # Convertim array-ul la lista pentru a putea fi folosit in DataFrame
                            proba_df = pd.DataFrame({
                                'Clasa': classes,
                                'Probabilitate': best_proba.tolist() if hasattr(best_proba, 'tolist') else best_proba
                            }).sort_values('Probabilitate', ascending=False)

                            st.write("Probabilitati pentru fiecare clasa:")
                            st.dataframe(proba_df, use_container_width=True)
                    except:
                        st.write("Modelul nu suporta afisarea probabilitatilor.")
                else:
                    best_pred = best_pipeline.predict(input_df)[0]
                    st.success(f"**Modelul cel mai bun ({best_model_name}) prezice: {best_pred:.4f}**")
    else:
        st.info(
            "Nu exista modele antrenate pentru a face predictii. Te rog antreneaza mai intai modelele in sectiunea 4.")

# Finalizare - mesaj daca nu sunt date incarcate
if df is None:

    st.info("Pentru a folosi functionalitatile ML, te rog incarca mai intai un fisier CSV sau Excel in sectiunea 1.")
