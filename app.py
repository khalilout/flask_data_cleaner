import os
import numpy as np
from flask import Flask, request, send_file
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

def safe_float(value):
    """Convertit en float ou None si NaN/inf"""
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

    # Lecture du fichier
    if ext in ['csv', 'txt']:
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    elif ext == 'json':
        df = pd.read_json(file)
    elif ext == 'xml':
        try:
            df = pd.read_xml(BytesIO(file.read()), xpath=".//record")
        except:
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
        return "Format non supporté", 400

    # ═══════════════════════════════════════════════════════════════════
    # ✅ COPIE DE L'ORIGINAL AVANT TOUT TRAITEMENT
    # ═══════════════════════════════════════════════════════════════════
    df_original = df.copy()

    # Liste des valeurs manquantes
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

    valeurs_manquantes_lower = [str(v).lower() for v in VALEURS_MANQUANTES if v]

    # ═══════════════════════════════════════════════════════════════════
    # ✅ CALCUL DES STATS SUR L'ORIGINAL (AVANT NETTOYAGE)
    # ═══════════════════════════════════════════════════════════════════

    # 1️⃣ VALEURS MANQUANTES PAR COLONNE
    stats_missing = {}
    for col in df_original.columns:
        nb_nan = int(df_original[col].isnull().sum())
        non_null = df_original[col].dropna()
        if len(non_null) > 0:
            nb_text = int(non_null.astype(str).str.strip().str.lower().isin(valeurs_manquantes_lower).sum())
        else:
            nb_text = 0
        stats_missing[col] = nb_nan + nb_text

    # 2️⃣ DOUBLONS TOTAUX (ligne complète identique)
    nb_doublons_total = int(df_original.duplicated(keep='first').sum())

    # 3️⃣ VALEURS ABERRANTES PAR COLONNE
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
            mask = (df_temp[col] < Q1 - 1.5 * IQR) | \
                   (df_temp[col] > Q3 + 1.5 * IQR) | \
                   (df_temp[col] < 0)
            stats_outliers[col] = int(mask.sum())
        else:
            stats_outliers[col] = 0

    for col in df_original.columns:
        if col not in stats_outliers:
            stats_outliers[col] = 0

    # ═══════════════════════════════════════════════════════════════════
    # ✅ NETTOYAGE DES DONNÉES
    # ═══════════════════════════════════════════════════════════════════

    # Remplacer les valeurs manquantes par NaN
    df.replace(VALEURS_MANQUANTES, np.nan, inplace=True)
    
    # ✅ SUPPRESSION DES DOUBLONS (lignes 100% identiques)
    df = df.drop_duplicates(keep='first')

    # Conversion numérique conditionnelle
    for col in df.columns:
        test_numeric = pd.to_numeric(df[col], errors='coerce')
        if test_numeric.notna().sum() / len(df) > 0.5:
            df[col] = test_numeric

    # Identifier colonnes numériques et catégorielles
    colonnes_numeriques = df.select_dtypes(include=["int64", "float64"]).columns
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns

    # Remplissage valeurs manquantes - NUMÉRIQUES
    for col in colonnes_numeriques:
        if df[col].isnull().sum() > 0:
            if df[col].notna().sum() > 0:
                skewness = df[col].skew()
                if abs(skewness) < 0.5:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(0, inplace=True)

    # Remplissage valeurs manquantes - CATÉGORIELLES
    for col in colonnes_categorielles:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "inconnu")

    # Traitement des valeurs aberrantes
    outlier_method = request.form.get("outlier_method", "none")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if outlier_method == "iqr":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = max(0, Q1 - 1.5*IQR)
            upper = Q3 + 1.5*IQR
            df[col] = df[col].clip(lower, upper)
            
    elif outlier_method == "median":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            med = df[col].median()
            mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR) | (df[col] < 0)
            df.loc[mask, col] = med
            
    elif outlier_method == "log":
        for col in num_cols:
            df[col] = df[col].clip(lower=0)
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])
                
    elif outlier_method == "delete":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= 0) & (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    # ═══════════════════════════════════════════════════════════════════
    # ✅ CONSTRUCTION DES STATS FINALES AVEC POURCENTAGES
    # ═══════════════════════════════════════════════════════════════════

    stats = {}
    num_cols_final = df.select_dtypes(include=["int64", "float64"]).columns
    total_rows = len(df_original)

    for col in df.columns:
        missing = stats_missing.get(col, 0)
        outliers = stats_outliers.get(col, 0)
        
        # Calcul des pourcentages
        missing_pct = round((missing / total_rows) * 100, 2) if total_rows > 0 else 0
        outliers_pct = round((outliers / total_rows) * 100, 2) if total_rows > 0 else 0
        duplicates_pct = round((nb_doublons_total / total_rows) * 100, 2) if total_rows > 0 else 0

        if col in num_cols_final:
            stats[col] = {
                "mean": safe_float(df[col].mean()),
                "median": safe_float(df[col].median()),
                "min": safe_float(df[col].min()),
                "max": safe_float(df[col].max()),
                "std": safe_float(df[col].std()),
                "missing": missing,
                "missing_pct": missing_pct,
                "duplicates": nb_doublons_total,
                "duplicates_pct": duplicates_pct,
                "outliers": outliers,
                "outliers_pct": outliers_pct
            }
        else:
            mode_val = "N/A"
            if len(df[col].mode()) > 0:
                mode_val = str(df[col].mode()[0])
            stats[col] = {
                "mode": mode_val,
                "unique_count": int(df[col].nunique()),
                "missing": missing,
                "missing_pct": missing_pct,
                "duplicates": nb_doublons_total,
                "duplicates_pct": duplicates_pct,
                "outliers": outliers,
                "outliers_pct": outliers_pct
            }

    # Export CSV
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