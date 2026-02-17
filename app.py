import os
import numpy as np
from flask import Flask, request, send_file, jsonify
import pandas as pd
import xmltodict
import io
import json

app = Flask(__name__)

@app.route("/")
def index():
    return "Service de nettoyage de données est opérationnel."

# Fusion intelligente des valeurs
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

# Détection automatique des colonnes clés
def detecter_colonnes_cles(df):
    colonnes_obj = df.select_dtypes(include=["object"]).columns
    colonnes_cles = []
    for col in colonnes_obj:
        if df[col].nunique() > 1:
            colonnes_cles.append(col)
    return colonnes_cles

# ✅ Convertit NaN/inf en None pour JSON valide
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

    # ── Lecture du fichier ──────────────────────────────────────────────
    if ext in ['csv', 'txt']:
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    elif ext == 'json':
        df = pd.read_json(file)
    elif ext == 'xml':
        content = file.read()
        data = xmltodict.parse(content)

        root_key = list(data.keys())[0]
        root = data[root_key]

        if isinstance(root, list):
            records = root
        elif isinstance(root, dict):
            records = None
            for key, value in root.items():
                if isinstance(value, list):
                    records = value
                    break
            if records is None:
                records = [root]
        else:
            records = [root]

        df = pd.DataFrame(records)
    else:
        return jsonify({"error": "Format non supporté"}), 400

    # ── LISTE DES VALEURS MANQUANTES ────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════
    # ✅✅✅  CALCUL DES STATS AVANT NETTOYAGE  ✅✅✅
    # ══════════════════════════════════════════════════════════════════

    # 1️⃣ Copie de l'original AVANT tout nettoyage
    df_original = df.copy()

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

    # ══════════════════════════════════════════════════════════════════
    # ✅✅✅  NETTOYAGE  ✅✅✅
    # ══════════════════════════════════════════════════════════════════

    # Remplacer les valeurs manquantes par NaN
    df.replace(VALEURS_MANQUANTES, np.nan, inplace=True)

    # Fusion des doublons
    colonnes_cles = detecter_colonnes_cles(df)
    if colonnes_cles:
        df = df.groupby(colonnes_cles, as_index=False).agg(fusion_valeurs)

    # Conversion numérique
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remplissage des valeurs manquantes
    colonnes_numeriques   = df.select_dtypes(include=["int64", "float64"]).columns
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns

    for col in colonnes_numeriques:
        if df[col].isnull().sum() > 0:
            skewness = df[col].skew()
            if abs(skewness) < 0.5:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    for col in colonnes_categorielles:
        if df[col].isnull().sum() > 0 and len(df[col].mode()) > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Traitement des valeurs aberrantes
    outlier_method = request.form.get("outlier_method", "none")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if outlier_method == "iqr":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            df[col] = df[col].clip(Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1))
    elif outlier_method == "median":
        for col in num_cols:
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            med = df[col].median()
            df.loc[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR), col] = med
    elif outlier_method == "log":
        for col in num_cols:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])
    elif outlier_method == "delete":
        for col in num_cols:
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df  = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # ══════════════════════════════════════════════════════════════════
    # ✅✅✅  CONSTRUCTION DES STATS FINALES  ✅✅✅
    # (statistiques descriptives sur données nettoyées
    #  + missing/duplicates/outliers de l'ORIGINAL)
    # ══════════════════════════════════════════════════════════════════

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

    # ── Export CSV ──────────────────────────────────────────────────────
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    response = send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='donnees_nettoyees.csv'
    )

    # ✅ JSON valide (pas de NaN)
    response.headers["X-Data-Stats"] = json.dumps(stats)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)