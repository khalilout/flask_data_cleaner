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
        df = pd.read_xml(BytesIO(file.read()))
    else:
        return "Format non supporté", 400
    # elif ext == 'xml':
    #     # Lire le contenu du fichier XML
    #     content = file.read()
    #     data = xmltodict.parse(content)
        
    #     # Trouver les données (première clé du dictionnaire)
    #     root_key = list(data.keys())[0]
    #     root = data[root_key]
        
    #     # Si c'est une liste, c'est bon
    #     if isinstance(root, list):
    #         records = root
    #     # Si c'est un dictionnaire
    #     elif isinstance(root, dict):
    #         # Chercher la première liste dans le dictionnaire
    #         records = None
    #         for key, value in root.items():
    #             if isinstance(value, list):
    #                 records = value
    #                 break
    #         # Si pas de liste trouvée, c'est un seul enregistrement
    #         if records is None:
    #             records = [root]
    #     else:
    #         # Si ce n'est ni liste ni dict, mettre dans une liste
    #         records = [root]
        
    #     # Créer le DataFrame
    #     df = pd.DataFrame(records) 
    # else:
    #     return "Format non supporté", 400


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
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')



    # Valeurs manquantes
    colonnes_numeriques = df.select_dtypes(include=["int64", "float64"]).columns #si la colonne est numérique
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns #si la colonne est catégorielle

    for col in colonnes_numeriques:
        if df[col].isnull().sum() > 0:
            skewness = df[col].skew()
            if abs(skewness) < 0.5:
                df[col] = df[col].fillna(df[col].mean())

            else:
                df[col] = df[col].fillna(df[col].median())

    for col in colonnes_categorielles:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

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

            

    # 🔹 Calcul des statistiques pour graphes
    stats = {}
    stats_avant = {}
    nb_aberrantes = {}
    nb_doublons_total = int(df.duplicated().sum())

    for col in df.columns:
        stats_avant[col] = {
            "missing": int(df[col].isnull().sum())
        }
        nb_aberrantes[col] = 0

    for col in df.columns:
        if col in num_cols:
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": float(df[col].std()),
                "missing": stats_avant.get(col, {}).get("missing", 0), 
                "duplicates": nb_doublons_total,  
                "outliers": nb_aberrantes.get(col, 0) 
            }
        else:
            stats[col] = {
                "mode": str(df[col].mode()[0]),
                "unique_count": int(df[col].nunique()),
                "missing": stats_avant.get(col, {}).get("missing", 0),
                "duplicates": nb_doublons_total,  
                "outliers": nb_aberrantes.get(col, 0) 
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