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


# ═══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def fusion_valeurs(series):
    """Fusion intelligente des valeurs dupliquées (version optimisée)."""
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


def detecter_colonnes_cles(df):
    """Détecte les colonnes texte à utiliser comme clés pour la fusion de doublons."""
    colonnes_obj = df.select_dtypes(include=["object"]).columns
    return [col for col in colonnes_obj if df[col].nunique() > 1]


def safe_float(value):
    """Convertit en float ou None si NaN/inf."""
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# CORRECTION 1 — Détection des anomalies de type dans les colonnes
# ═══════════════════════════════════════════════════════════════════

def detecter_anomalies_type(df_original):
    """
    Détecte deux cas que l'IQR seul ne couvre pas :
      • Valeur NUMÉRIQUE dans une colonne CATÉGORIELLE  → ex: OWN_OCCUPIED="12"
      • Valeur TEXTE non numérique dans une colonne NUMÉRIQUE → ex: NUM_BATH="HURLEY"

    Retourne un dict {colonne: nb_anomalies} et la liste des cellules concernées
    pour le logging.
    """
    anomalies = {}
    details = {}

    for col in df_original.columns:
        serie = df_original[col].dropna().astype(str).str.strip()

        # Tente de convertir chaque valeur en nombre
        as_numeric = pd.to_numeric(serie, errors='coerce')
        pct_numeric = as_numeric.notna().sum() / len(serie) if len(serie) > 0 else 0

        if pct_numeric <= 0.5:
            # Colonne CATÉGORIELLE → on cherche les valeurs qui sont des nombres
            masque_num = as_numeric.notna()
            nb = int(masque_num.sum())
            if nb > 0:
                anomalies[col] = anomalies.get(col, 0) + nb
                details[col] = details.get(col, [])
                details[col] += [
                    f"valeur numérique dans col. catégorielle : '{v}'"
                    for v in serie[masque_num].tolist()
                ]
        else:
            # Colonne NUMÉRIQUE → on cherche les valeurs qui ne sont PAS des nombres
            masque_txt = as_numeric.isna()
            nb = int(masque_txt.sum())
            if nb > 0:
                anomalies[col] = anomalies.get(col, 0) + nb
                details[col] = details.get(col, [])
                details[col] += [
                    f"texte invalide dans col. numérique : '{v}'"
                    for v in serie[masque_txt].tolist()
                ]

    return anomalies, details


# ═══════════════════════════════════════════════════════════════════
# CORRECTION 2 — Nettoyage des anomalies de type
# ═══════════════════════════════════════════════════════════════════

def corriger_anomalies_type(df):
    """
    Corrige les anomalies détectées :
      • Valeur numérique dans colonne catégorielle → remplacée par NaN
        (sera ensuite remplie par le mode de la colonne)
      • Valeur texte non numérique dans colonne numérique → remplacée par NaN
        (sera ensuite remplie par moyenne ou médiane)
    """
    for col in df.columns:
        serie = df[col].dropna().astype(str).str.strip()
        as_numeric = pd.to_numeric(serie, errors='coerce')
        pct_numeric = as_numeric.notna().sum() / len(serie) if len(serie) > 0 else 0

        if pct_numeric <= 0.5:
            # Colonne catégorielle : remplacer les valeurs numériques par NaN
            def remplacer_num_dans_cat(val):
                if pd.isna(val):
                    return val
                try:
                    float(str(val).strip())
                    return np.nan  # c'est un nombre → anomalie → NaN
                except ValueError:
                    return val
            df[col] = df[col].apply(remplacer_num_dans_cat)

        else:
            # Colonne numérique : remplacer les textes invalides par NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ═══════════════════════════════════════════════════════════════════
# ROUTE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════

@app.route('/clean', methods=['POST'])
@app.route('/api/clean', methods=['POST'])
def import_file():
    file = request.files['file']
    ext = file.filename.split('.')[-1].lower()

    # ───────────────────────────────────────────────────────────────
    # LECTURE DU FICHIER
    # ───────────────────────────────────────────────────────────────
    if ext in ['csv', 'txt']:
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    elif ext == 'json':
        df = pd.read_json(file)
    elif ext == 'xml':
        try:
            df = pd.read_xml(BytesIO(file.read()), xpath=".//record")
        except Exception:
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

    # ───────────────────────────────────────────────────────────────
    # COPIE DE L'ORIGINAL AVANT TOUT TRAITEMENT
    # ───────────────────────────────────────────────────────────────
    df_original = df.copy()

    # Liste des valeurs manquantes textuelles
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

    # ───────────────────────────────────────────────────────────────
    # STATS AVANT NETTOYAGE — 1. VALEURS MANQUANTES
    # ───────────────────────────────────────────────────────────────
    stats_missing = {}
    for col in df_original.columns:
        nb_nan = int(df_original[col].isnull().sum())
        non_null = df_original[col].dropna()
        if len(non_null) > 0:
            nb_text = int(
                non_null.astype(str).str.strip().str.lower()
                .isin(valeurs_manquantes_lower).sum()
            )
        else:
            nb_text = 0
        stats_missing[col] = nb_nan + nb_text

    # ───────────────────────────────────────────────────────────────
    # STATS AVANT NETTOYAGE — 2. DOUBLONS
    # ───────────────────────────────────────────────────────────────
    nb_doublons_total = int(df_original.duplicated().sum())

    # ───────────────────────────────────────────────────────────────
    # STATS AVANT NETTOYAGE — 3. OUTLIERS IQR + négatifs
    # ───────────────────────────────────────────────────────────────
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
            mask = (
                (df_temp[col] < Q1 - 1.5 * IQR) |
                (df_temp[col] > Q3 + 1.5 * IQR) |
                (df_temp[col] < 0)
            )
            stats_outliers[col] = int(mask.sum())
        else:
            stats_outliers[col] = 0

    for col in df_original.columns:
        if col not in stats_outliers:
            stats_outliers[col] = 0

    # ───────────────────────────────────────────────────────────────
    # STATS AVANT NETTOYAGE — 4. ANOMALIES DE TYPE (NOUVEAU ✅)
    # Détecte OWN_OCCUPIED="12" et NUM_BATH="HURLEY"
    # ───────────────────────────────────────────────────────────────
    anomalies_type, details_anomalies = detecter_anomalies_type(df_original)

    # On ajoute les anomalies de type aux outliers pour l'affichage frontend
    for col, nb in anomalies_type.items():
        stats_outliers[col] = stats_outliers.get(col, 0) + nb

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 1 : valeurs manquantes textuelles → NaN
    # ───────────────────────────────────────────────────────────────
    df.replace(VALEURS_MANQUANTES, np.nan, inplace=True)

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 2 (NOUVEAU ✅) : corriger anomalies de type
    # OWN_OCCUPIED="12" → NaN  |  NUM_BATH="HURLEY" → NaN
    # ───────────────────────────────────────────────────────────────
    df = corriger_anomalies_type(df)

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 3 : fusion/suppression des doublons
    # ───────────────────────────────────────────────────────────────
    colonnes_cles = detecter_colonnes_cles(df)
    if colonnes_cles and len(df) < 2000:
        df = df.groupby(colonnes_cles, as_index=False).agg(fusion_valeurs)
    elif len(df) >= 2000:
        df = df.drop_duplicates(keep='first')

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 4 : conversion des colonnes numériques
    # (seulement si > 50 % de valeurs numériques valides)
    # ───────────────────────────────────────────────────────────────
    for col in df.columns:
        test_numeric = pd.to_numeric(df[col], errors='coerce')
        if len(df) > 0 and test_numeric.notna().sum() / len(df) > 0.5:
            df[col] = test_numeric

    colonnes_numeriques   = df.select_dtypes(include=["int64", "float64"]).columns
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 5 : remplissage des NaN numériques
    # ───────────────────────────────────────────────────────────────
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

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 6 : remplissage des NaN catégoriels
    # ───────────────────────────────────────────────────────────────
    for col in colonnes_categorielles:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "inconnu")

    # ───────────────────────────────────────────────────────────────
    # NETTOYAGE — Étape 7 : traitement des outliers IQR
    # ───────────────────────────────────────────────────────────────
    outlier_method = request.form.get("outlier_method", "none")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if outlier_method == "iqr":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = max(0, Q1 - 1.5 * IQR)
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

    elif outlier_method == "median":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            med = df[col].median()
            mask = (
                (df[col] < Q1 - 1.5 * IQR) |
                (df[col] > Q3 + 1.5 * IQR) |
                (df[col] < 0)
            )
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
            df = df[
                (df[col] >= 0) &
                (df[col] >= Q1 - 1.5 * IQR) &
                (df[col] <= Q3 + 1.5 * IQR)
            ]

    # ───────────────────────────────────────────────────────────────
    # CONSTRUCTION DES STATS FINALES
    # ───────────────────────────────────────────────────────────────
    stats = {}
    num_cols_final = df.select_dtypes(include=["int64", "float64"]).columns

    for col in df.columns:
        missing  = stats_missing.get(col, 0)
        outliers = stats_outliers.get(col, 0)

        if col in num_cols_final:
            stats[col] = {
                "mean":       safe_float(df[col].mean()),
                "median":     safe_float(df[col].median()),
                "min":        safe_float(df[col].min()),
                "max":        safe_float(df[col].max()),
                "std":        safe_float(df[col].std()),
                "missing":    missing,
                "duplicates": nb_doublons_total,
                "outliers":   outliers,
                # NOUVEAU : anomalies de type comptabilisées séparément
                "type_anomalies": anomalies_type.get(col, 0),
            }
        else:
            mode_val = "N/A"
            if len(df[col].mode()) > 0:
                mode_val = str(df[col].mode()[0])
            stats[col] = {
                "mode":         mode_val,
                "unique_count": int(df[col].nunique()),
                "missing":      missing,
                "duplicates":   nb_doublons_total,
                "outliers":     outliers,
                # NOUVEAU : anomalies de type comptabilisées séparément
                "type_anomalies": anomalies_type.get(col, 0),
            }

    # ───────────────────────────────────────────────────────────────
    # EXPORT CSV
    # ───────────────────────────────────────────────────────────────
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