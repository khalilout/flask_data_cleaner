import os
import numpy as np
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

# ✅ MODIFICATION 1 : Fusion optimisée
def fusion_valeurs(series):
    """Version optimisée pour éviter les timeouts"""
    vals = series.tolist()
    cleaned = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    
    if len(cleaned) == 0:
        return np.nan
    elif len(cleaned) == 1:
        return cleaned[0]
    else:
        return " ; ".join(cleaned)
    
# Détection automatique des colonnes clés
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

    VALEURS_MANQUANTES = [
        "", " ", "  ", "   ", "\t", "\n", "\r",
        "--", "---", "—", "–", "-", "_",
        "?", "??", "***", "*",
        "NA", "N/A", "n/a", "na", "Na",
        "NULL", "null", "None", "none", "Nil", "nil",
        "Missing", "missing", "Unknown", "unknown",
        "Undefined", "undefined",
        "Not Available", "not available",
        "Not Applicable", "not applicable",
        "#N/A", "#NA", "#VALUE!", "#DIV/0!",
        "#REF!", "#NAME?", "#NUM!",
        "vide", "inconnu", "non renseigné", "non disponible", "aucun",
        "empty", "not provided", "no data"
    ]

    df.replace(VALEURS_MANQUANTES, np.nan, inplace=True)
    
    # ✅ MODIFICATION 2 : Fusion conditionnelle
    colonnes_cles = detecter_colonnes_cles(df)
    if colonnes_cles and len(df) < 2000:
        df = df.groupby(colonnes_cles, as_index=False).agg(fusion_valeurs)
    elif len(df) >= 2000:
        df = df.drop_duplicates(keep='first')

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

    colonnes_numeriques = df.select_dtypes(include=["int64", "float64"]).columns
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns

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

    stats_missing = {}
    for col in df_original.columns:
        nb_nan = df_original[col].isnull().sum()
        nb_liste = df_original[col].isin(VALEURS_MANQUANTES).sum()
        stats_missing[col] = int(nb_nan + nb_liste)

    nb_doublons_total = int(df_original.duplicated().sum())

    df_temp = df_original.copy()
    for col in df_temp.columns:
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    cols_num_original = df_temp.select_dtypes(include=["int64", "float64"]).columns

    stats_outliers = {}
    for col in cols_num_original:
        serie = df_temp[col].dropna()
        if len(serie) > 0:
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            mask = (df_temp[col] < Q1 - 1.5 * IQR) | (df_temp[col] > Q3 + 1.5 * IQR)
            stats_outliers[col] = int(mask.sum())
        else:
            stats_outliers[col] = 0

    stats = {}
    num_cols_final = df.select_dtypes(include=["int64", "float64"]).columns

    for col in df.columns:
        missing = stats_missing.get(col, 0)
        outliers = stats_outliers.get(col, 0)

        if col in num_cols_final:
            stats[col] = {
                "mean": safe_float(df[col].mean()),
                "median": safe_float(df[col].median()),
                "min": safe_float(df[col].min()),
                "max": safe_float(df[col].max()),
                "std": safe_float(df[col].std()),
                "missing": missing,
                "duplicates": nb_doublons_total,
                "outliers": outliers
            }
        else:
            mode_val = "N/A"
            if len(df[col].mode()) > 0:
                mode_val = str(df[col].mode()[0])
            stats[col] = {
                "mode": mode_val,
                "unique_count": int(df[col].nunique()),
                "missing": missing,
                "duplicates": nb_doublons_total,
                "outliers": 0
            }

    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    response = send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='donnees_nettoyees.csv'
    )

    response.headers["X-Data-Stats"] = json.dumps(stats)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))