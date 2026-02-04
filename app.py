import os
import numpy as np  # ajouté
from flask import Flask, request, send_file, make_response
import pandas as pd
import xmltodict
import io
import json

app = Flask(__name__)

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
        data = xmltodict.parse(file.read())
        root = list(data.values())[0]        
        records = list(root.values())[0]     
        if isinstance(records, dict):
            records = [records]              
        df = pd.DataFrame(records)  
    else:
        return "Format non supporté", 400


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
        df[col] = pd.to_numeric(df[col], errors='ignore')



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
            df[col].fillna(df[col].mode()[0], inplace=True)

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
    for col in df.columns:
        if col in num_cols:
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": float(df[col].std()),
            }
        else:
            stats[col] = {
                "mode": str(df[col].mode()[0]),
                "unique_count": int(df[col].nunique())
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