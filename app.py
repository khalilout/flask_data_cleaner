import os
import numpy as np  # ajouté
from flask import Flask, request, send_file, make_response
import pandas as pd
import xmltodict
import io
import json
from io import BytesIO 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    return "Service de nettoyage de données est opérationnel."

#  Fusion intelligente des valeurs
def fusion_valeurs(series):
    valeurs = series.dropna().astype(str)
    valeurs = valeurs[valeurs.str.strip() != ""]
    uniques = valeurs.unique()

    if len(uniques) == 0:
        return np.nan
    elif len(uniques) == 1:
        return uniques[0]
    else:
        return " ; ".join(uniques)
    
#  Détection automatique des colonnes clés
def detecter_colonnes_cles(df):
    colonnes_obj = df.select_dtypes(include=["object"]).columns
    colonnes_cles = []

    for col in colonnes_obj:
        if df[col].nunique() > 1:
            colonnes_cles.append(col)

    return colonnes_cles

def safe_float(value):
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
    except:
        return None



@app.route('/clean', methods=['POST'])
@app.route('/api/clean', methods=['POST'])
def import_file():
    file = request.files['file']
    ext = file.filename.split('.')[-1].lower()

    # Lecture du fichier selon son format
    if ext in ['csv', 'txt']:
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    elif ext == 'json':
        df = pd.read_json(file)
    elif ext == 'xml':
        df = pd.read_xml(BytesIO(file.read()), xpath=".//record")
    else:
        return "Format non supporté", 400

    df_original = df.copy()

    #  Nettoyage initial ("--" → NaN)
    VALEURS_MANQUANTES = [
        "", " ", "  ", "   ", "\t", "\n", "\r",

        "--", "---", "—", "–", "-", "_",
        "?", "??", "***", "*",

        "NA", "N/A", "n/a", "na", "Na",
        "NULL", "null",
        "None", "none",
        "Nil", "nil",

        "Missing", "missing",
        "Unknown", "unknown",
        "Undefined", "undefined",

        "Not Available", "not available",
        "Not Applicable", "not applicable",

        "#N/A", "#NA", "#VALUE!", "#DIV/0!",
        "#REF!", "#NAME?", "#NUM!",

        "vide", "inconnu",
        "non renseigné", "non disponible",
        "aucun",

        "empty", "not provided", "no data"
    ]

    df.replace(VALEURS_MANQUANTES, np.nan, inplace=True)
    

    #  FUSION DES DOUBLONS (IMPORTANT)
    colonnes_cles = detecter_colonnes_cles(df)
    if colonnes_cles:
        df = df.groupby(colonnes_cles, as_index=False).agg(fusion_valeurs)

    # Conversion numérique
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

        if df[col].notna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        df[col] = df[col].fillna("inconnu")




    # Valeurs manquantes
    colonnes_numeriques = df.select_dtypes(include=["int64", "float64"]).columns #si la colonne est numérique
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns #si la colonne est catégorielle

    for col in colonnes_numeriques:
        if df[col].isnull().sum() > 0:
            skewness = df[col].skew()
            if abs(skewness) < 0.5:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    for col in colonnes_categorielles:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "inconnu")

    # Valeur Aberante
    outlier_method = request.form.get("outlier_method", "none")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if outlier_method == "iqr":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
    elif outlier_method == "median":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            med = df[col].median()
            df.loc[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR), col] = med
    elif outlier_method == "log":
        for col in num_cols:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])
    elif outlier_method == "delete":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]


      # 1️⃣ Copie de l'original AVANT tout nettoyage
    

    # 2️⃣ Valeurs manquantes : NaN + valeurs de la liste
    stats_missing = {}
    for col in df_original.columns:
        nb_nan   = df_original[col].isnull().sum()
        nb_liste = df_original[col].isin(VALEURS_MANQUANTES).sum()
        stats_missing[col] = int(nb_nan + nb_liste)

    # 3️⃣ Doublons : nombre de lignes en double sur le fichier entier
    nb_doublons_total = int(df_original.duplicated().sum())

    # 4️⃣ Valeurs aberrantes : calculées sur les colonnes numériques de l'original
    #    On convertit temporairement pour détecter les numériques
    df_temp = df_original.copy()
    for col in df_temp.columns:
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    cols_num_original = df_temp.select_dtypes(include=["int64", "float64"]).columns

    stats_outliers = {}
    for col in cols_num_original:
        serie = df_temp[col].dropna()
        if len(serie) > 0:
            Q1  = serie.quantile(0.25)
            Q3  = serie.quantile(0.75)
            IQR = Q3 - Q1
            mask = (df_temp[col] < Q1 - 1.5 * IQR) | (df_temp[col] > Q3 + 1.5 * IQR)
            stats_outliers[col] = int(mask.sum())
        else:
            stats_outliers[col] = 0
            

    # # 🔹 Calcul des statistiques pour graphes
    # stats = {}
    # stats_avant = {}
    # nb_aberrantes = {}
    # nb_doublons_total = int(df.duplicated().sum())

    # for col in df.columns:
    #     stats_avant[col] = {
    #         "missing": int(df[col].isnull().sum())
    #     }
    #     nb_aberrantes[col] = 0

    # for col in df.columns:
    #     if col in num_cols:
    #         stats[col] = {
    #             "mean": float(df[col].mean()),
    #             "median": float(df[col].median()),
    #             "min": float(df[col].min()),
    #             "max": float(df[col].max()),
    #             "std": float(df[col].std()),
    #             "missing": stats_avant.get(col, {}).get("missing", 0), 
    #             "duplicates": nb_doublons_total,  
    #             "outliers": nb_aberrantes.get(col, 0) 
    #         }
    #     else:
    #         stats[col] = {
    #             "mode": str(df[col].mode().iloc[0] if not df[col].mode().empty else "inconnu"),
    #             "unique_count": int(df[col].nunique()),
    #             "missing": stats_avant.get(col, {}).get("missing", 0),
    #             "duplicates": nb_doublons_total,  
    #             "outliers": nb_aberrantes.get(col, 0) 
    #         }


    stats = {}
    num_cols_final = df.select_dtypes(include=["int64", "float64"]).columns

    for col in df.columns:
        missing   = stats_missing.get(col, 0)
        outliers  = stats_outliers.get(col, 0)

        if col in num_cols_final:
            stats[col] = {
                "mean"      : safe_float(df[col].mean()),
                "median"    : safe_float(df[col].median()),
                "min"       : safe_float(df[col].min()),
                "max"       : safe_float(df[col].max()),
                "std"       : safe_float(df[col].std()),
                "missing"   : missing,            # ✅ AVANT nettoyage
                "duplicates": nb_doublons_total,  # ✅ AVANT nettoyage
                "outliers"  : outliers            # ✅ AVANT nettoyage
            }
        else:
            mode_val = "N/A"
            if len(df[col].mode()) > 0:
                mode_val = str(df[col].mode()[0])
            stats[col] = {
                "mode"        : mode_val,
                "unique_count": int(df[col].nunique()),
                "missing"     : missing,            # ✅ AVANT nettoyage
                "duplicates"  : nb_doublons_total,  # ✅ AVANT nettoyage
                "outliers"    : 0
            }


    # Export CSV
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Retour CSV avec stats dans headers (Laravel pourra les lire)
    response = send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='donnees_nettoyees.csv'
    )

    # Ajouter stats dans headers (JSON stringifiée)
    import json
    response.headers["X-Data-Stats"] = json.dumps(stats)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    # app.run(debug=True, port=5000)