"""
Moteur d'estimation de Pain (Delta Infinity) — version étendue.

Couvre les méthodes QUANTITATIVES du catalogue Pain :
  mco, mcg, logit, probit, gmm (via IV/2SLS), panel (fixe/aléatoire), ardl, sem

Ne couvre PAS (volontairement, voir note en bas) :
  plssem, thematique, contenu
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # pas d'écran sur un serveur, indispensable sur Render
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def _nettoyer_nan(obj):
    """Remplace toute valeur NaN/Infinity par null (None) avant de répondre.
    Python écrit NaN en toutes lettres dans le JSON par défaut (ex. médiane
    d'un groupe entièrement vide) — valide en Python, mais pas en JSON
    standard, donc le navigateur refuse de le lire. On assainit récursivement
    toutes les réponses pour ne plus jamais renvoyer ça."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nettoyer_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nettoyer_nan(v) for v in obj]
    return obj


def jsonify_propre(obj, status=200):
    return jsonify(_nettoyer_nan(obj)), status


def _clean_dataframe(data, colonnes):
    df = pd.DataFrame(data)
    for c in colonnes:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans les données : {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=colonnes)


def _coeffs_depuis_fit(fit, noms):
    return [
        {
            "variable": nom,
            "coefficient": round(float(fit.params[nom]), 4),
            "erreur_standard": round(float(fit.bse[nom]), 4),
            "p_value": round(float(fit.pvalues[nom]), 4),
            "significatif_5pct": bool(fit.pvalues[nom] < 0.05),
        }
        for nom in noms
    ]


def _mco(df, dependante, independantes):
    X = sm.add_constant(df[independantes]); y = df[dependante]
    fit = sm.OLS(y, X).fit()
    return {
        "methode": "MCO (moindres carrés ordinaires)",
        "n": int(fit.nobs), "r2": round(float(fit.rsquared), 4), "r2_ajuste": round(float(fit.rsquared_adj), 4),
        "coefficients": _coeffs_depuis_fit(fit, ["const"] + independantes),
        "figures": _figures_diagnostic(fit, df, dependante, independantes),
    }


def _mcg(df, dependante, independantes):
    # FGLS en 2 étapes : corrige l'hétéroscédasticité en pondérant par
    # l'inverse de la variance estimée des résidus MCO.
    X = sm.add_constant(df[independantes]); y = df[dependante]
    ols_fit = sm.OLS(y, X).fit()
    resid2 = ols_fit.resid ** 2
    aux = sm.OLS(np.log(resid2 + 1e-8), X).fit()
    weights = 1.0 / np.exp(aux.fittedvalues)
    fit = sm.WLS(y, X, weights=weights).fit()
    return {
        "methode": "MCG (moindres carrés généralisés, FGLS 2 étapes)",
        "n": int(fit.nobs), "r2": round(float(fit.rsquared), 4), "r2_ajuste": round(float(fit.rsquared_adj), 4),
        "coefficients": _coeffs_depuis_fit(fit, ["const"] + independantes),
        "note": "Pondération estimée automatiquement à partir de la variance des résidus MCO (FGLS en 2 étapes).",
    }


def _logit_probit(df, dependante, independantes, modele):
    y = df[dependante]
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"La variable dépendante doit être binaire (0/1) pour un {modele}.")
    X = sm.add_constant(df[independantes])
    fit = (sm.Logit(y, X) if modele == "Logit" else sm.Probit(y, X)).fit(disp=0)
    return {
        "methode": modele, "n": int(fit.nobs), "pseudo_r2": round(float(fit.prsquared), 4),
        "coefficients": _coeffs_depuis_fit(fit, ["const"] + independantes),
        "figures": _figures_diagnostic(fit, df, dependante, independantes),
    }


def _gmm(df, dependante, independantes, instruments):
    # GMM implémenté via 2SLS (variables instrumentales) — le cas d'usage
    # le plus courant en pratique pour corriger l'endogénéité.
    if not instruments:
        raise ValueError("GMM (variables instrumentales) nécessite au moins un 'instruments' (liste de noms de colonnes).")
    from linearmodels.iv import IV2SLS
    colonnes = [dependante] + independantes + instruments
    df2 = _clean_dataframe(df.to_dict("records"), colonnes)
    endogenes = [v for v in independantes if v in instruments] or independantes[:1]
    exogenes = [v for v in independantes if v not in endogenes]
    fit = IV2SLS(df2[dependante], sm.add_constant(df2[exogenes]) if exogenes else None, df2[endogenes], df2[instruments]).fit()
    return {
        "methode": "GMM / Variables instrumentales (2SLS)", "n": int(fit.nobs),
        "coefficients": [
            {"variable": nom, "coefficient": round(float(fit.params[nom]), 4), "erreur_standard": round(float(fit.std_errors[nom]), 4), "p_value": round(float(fit.pvalues[nom]), 4), "significatif_5pct": bool(fit.pvalues[nom] < 0.05)}
            for nom in fit.params.index
        ],
        "note": "GMM implémenté ici via 2SLS (cas le plus courant en pratique). Un vrai GMM multi-moments généralisé pourra être ajouté plus tard si un cas précis l'exige.",
    }


def _panel(df, dependante, independantes, entite, periode, effets):
    if not entite or not periode:
        raise ValueError("Panel nécessite 'entite' et 'periode' (noms de colonnes identifiant individu et temps).")
    from linearmodels.panel import PanelOLS, RandomEffects
    colonnes = [dependante] + independantes + [entite, periode]
    df2 = _clean_dataframe(df.to_dict("records"), colonnes)

    notes_exclusion = []

    # 1. Un pays avec une seule année dans l'échantillon n'apporte aucune
    # variation utile une fois les effets fixes retirés.
    effectifs = df2[entite].value_counts()
    pays_singleton = effectifs[effectifs < 2].index.tolist()
    if pays_singleton:
        df2 = df2[~df2[entite].isin(pays_singleton)]
        notes_exclusion.append(f"Pays exclu(s) (une seule année disponible) : {', '.join(pays_singleton)}.")

    # 2. Un pays qui a 0% de données ORIGINALES pour une variable devient,
    # après imputation par médiane, CONSTANT sur toute sa période pour
    # cette variable — une valeur fabriquée, identique chaque année. Une
    # fois les effets fixes retirés, ce pays contribue alors une colonne de
    # zéros exacts pour cette variable, ce qui peut casser le rang de la
    # matrice explicative ("exog does not have full column rank"). On
    # retire ces pays PAR VARIABLE concernée plutôt que de risquer une
    # estimation numériquement instable ou un plantage.
    for var in independantes:
        variance_par_pays = df2.groupby(entite)[var].nunique()
        pays_constants = variance_par_pays[variance_par_pays <= 1].index.tolist()
        if pays_constants and len(pays_constants) < df2[entite].nunique():  # jamais vider tout le panel
            df2 = df2[~df2[entite].isin(pays_constants)]
            notes_exclusion.append(f"Pays exclu(s) pour '{var}' (aucune donnée d'origine, valeur imputée constante sur toute la période — pas de variation exploitable avec effets fixes) : {', '.join(pays_constants)}.")

    note_pays_exclus = " ".join(notes_exclusion) if notes_exclusion else None

    df2 = df2.set_index([entite, periode])
    X = sm.add_constant(df2[independantes])
    note_rang = None
    if effets == "aleatoire":
        fit = RandomEffects(df2[dependante], X).fit()
        nom_methode = "Panel (effets aléatoires)"
    else:
        try:
            fit = PanelOLS(df2[dependante], X, entity_effects=True).fit()
        except ValueError as e:
            if "full column rank" not in str(e):
                raise
            # Dernier recours : une colinéarité reste malgré les exclusions
            # ci-dessus (ex. deux variables imputées qui varient toujours
            # exactement ensemble). On le signale sans cacher le problème
            # plutôt que de planter, mais les coefficients concernés sont à
            # interpréter avec prudence.
            fit = PanelOLS(df2[dependante], X, entity_effects=True, check_rank=False).fit()
            note_rang = "Colinéarité parfaite détectée entre certaines variables explicatives même après retrait des pays à une seule année — l'estimation a quand même été calculée, mais certains coefficients peuvent être numériquement instables. Vérifie si deux variables de contrôle varient toujours ensemble."
        nom_methode = "Panel (effets fixes)"
    resultat = {
        "methode": nom_methode, "n": int(fit.nobs), "r2": round(float(fit.rsquared), 4),
        "coefficients": [
            {"variable": nom, "coefficient": round(float(fit.params[nom]), 4), "erreur_standard": round(float(fit.std_errors[nom]), 4), "p_value": round(float(fit.pvalues[nom]), 4), "significatif_5pct": bool(fit.pvalues[nom] < 0.05)}
            for nom in fit.params.index
        ],
    }
    if note_pays_exclus: resultat["note_pays_exclus"] = note_pays_exclus
    if note_rang: resultat["note_colinearite"] = note_rang
    return resultat


def _ardl(df, dependante, independantes, ar_lags, dl_lags):
    from statsmodels.tsa.ardl import ARDL
    colonnes = [dependante] + independantes
    df2 = _clean_dataframe(df.to_dict("records"), colonnes).reset_index(drop=True)
    lags_dict = {v: dl_lags or 1 for v in independantes}
    fit = ARDL(df2[dependante], lags=ar_lags or 1, exog=df2[independantes], order=lags_dict).fit()
    noms = list(fit.params.index)
    return {
        "methode": f"ARDL({ar_lags or 1})", "n": int(fit.nobs),
        "coefficients": [
            {"variable": nom, "coefficient": round(float(fit.params[nom]), 4), "erreur_standard": round(float(fit.bse[nom]), 4), "p_value": round(float(fit.pvalues[nom]), 4), "significatif_5pct": bool(fit.pvalues[nom] < 0.05)}
            for nom in noms
        ],
        "note": "AR lags et DL lags par défaut à 1 si non précisés — à ajuster selon les critères d'information (AIC/BIC) pour un vrai travail académique.",
    }


def _sem(data, modele_sem):
    if not modele_sem:
        raise ValueError("SEM nécessite 'modele_sem' : la spécification en syntaxe lavaan, ex. \"Y ~ X1 + X2\".")
    import semopy
    df = pd.DataFrame(data)
    model = semopy.Model(modele_sem)
    model.fit(df)
    resultats = model.inspect()
    return {
        "methode": "SEM (équations structurelles)", "n": len(df),
        "coefficients": [
            {"chemin": f"{r['lval']} {r['op']} {r['rval']}", "estimation": round(float(r["Estimate"]), 4), "p_value": (round(float(r["p-value"]), 4) if pd.notna(r.get("p-value")) else None)}
            for _, r in resultats.iterrows()
        ],
    }


def _nettoyer_serie(data, colonnes_valeurs, periode_col="annee", seuil_manquant=0.4, colonnes_protegees=None, entite_col=None):
    """Nettoyage déterministe et réutilisable (pas généré à la volée) :
    - aligne les séries sur la période commune
    - interpole les valeurs manquantes isolées (au maximum 1 an d'écart)
    - remplace tout ce qui reste manquant par la MÉDIANE de la variable
      (par entité si entite_col est fourni — ex. par pays en panel — sinon
      médiane globale), plutôt que de supprimer la ligne : perdre une année
      entière d'observations coûte souvent plus cher scientifiquement
      qu'une imputation transparente et signalée
    - signale (sans les supprimer automatiquement) les valeurs aberrantes
      au-delà de 3 écarts-types, pour rester transparent plutôt que de
      décider seul de jeter une observation légitime

    colonnes_protegees (dépendante + variable(s) d'intérêt) ne sont JAMAIS
    rejetées pour cause de trop de données manquantes — les rejeter rend
    l'estimation soit impossible (dépendante) soit vide de sens par rapport
    à l'hypothèse testée (variable d'intérêt). Seules les variables de
    contrôle peuvent être écartées si trop incomplètes.
    """
    df = pd.DataFrame(data)
    if periode_col not in df.columns:
        raise ValueError(f"Colonne de période manquante : {periode_col}")
    df = df.sort_values(periode_col)
    for c in colonnes_valeurs:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    protegees = set(colonnes_protegees or [colonnes_valeurs[0]])
    controles = [c for c in colonnes_valeurs if c not in protegees]
    taux_manquant = df[colonnes_valeurs].isna().mean()
    colonnes_ok = [c for c in colonnes_valeurs if c in protegees] + [c for c in controles if taux_manquant[c] <= seuil_manquant]
    colonnes_rejetees = [c for c in controles if c not in colonnes_ok]
    taux_protegees = {c: round(float(taux_manquant[c]), 3) for c in protegees if c in taux_manquant}

    df[colonnes_ok] = df[colonnes_ok].interpolate(method="linear", limit=1, limit_direction="both")

    # Si une entité (ex. un pays) a 0% de données D'ORIGINE pour une
    # variable retenue, on l'exclut directement ici plutôt que de lui
    # fabriquer une valeur constante par médiane globale — cette valeur
    # n'aurait aucune variation réelle et fausserait toute estimation à
    # effets fixes par la suite. On ne le fait jamais si ça viderait tout
    # le jeu de données (variable protégée avec un seul pays disponible,
    # par exemple).
    entites_exclues = {}
    if entite_col and entite_col in df.columns:
        for c in colonnes_ok:
            manquant_par_entite = df.groupby(entite_col)[c].apply(lambda s: s.isna().all())
            vides = manquant_par_entite[manquant_par_entite].index.tolist()
            if vides and len(vides) < df[entite_col].nunique():
                df = df[~df[entite_col].isin(vides)]
                entites_exclues[c] = vides

    imputations = {}
    for c in colonnes_ok:
        manquants_avant = int(df[c].isna().sum())
        if manquants_avant == 0:
            continue
        if entite_col and entite_col in df.columns:
            medianes = df.groupby(entite_col)[c].transform("median")
            df[c] = df[c].fillna(medianes)
        # Repli global si toujours manquant (ex. une entité n'a AUCUNE valeur)
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
        imputations[c] = manquants_avant

    df_propre = df.dropna(subset=colonnes_ok)  # ne devrait plus rien retirer, sauf médiane elle-même indéfinie (colonne 100% vide)

    aberrantes = {}
    for c in colonnes_ok:
        z = (df_propre[c] - df_propre[c].mean()) / (df_propre[c].std() or 1)
        idx = df_propre.index[z.abs() > 3]
        if len(idx) > 0:
            aberrantes[c] = df_propre.loc[idx, [periode_col, c]].to_dict("records")

    return {
        "data": df_propre.to_dict("records"),
        "n_avant": len(df), "n_apres": len(df_propre),
        "colonnes_rejetees": colonnes_rejetees,
        "taux_manquant_variables_protegees": taux_protegees,
        "valeurs_imputees_par_mediane": imputations,
        "entites_exclues_donnees_absentes": entites_exclues,
        "valeurs_aberrantes_signalees": aberrantes,
    }


@app.route("/nettoyer", methods=["POST"])
def nettoyer():
    try:
        p = request.get_json(force=True)
        resultat = _nettoyer_serie(p.get("data", []), p.get("colonnes", []), p.get("periode_col", "annee"), p.get("seuil_manquant", 0.4), p.get("colonnes_protegees"), p.get("entite_col"))
        return jsonify_propre(resultat)
    except Exception as e:
        return jsonify_propre({"erreur": str(e)}, 500)


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def _figures_diagnostic(fit, df, dependante, independantes):
    """Reprend l'idée de DeepSeek (heatmap, QQ-plot, résidus) — utile
    seulement pour les modèles à résidus classiques (mco/mcg/logit/probit)."""
    figures = []
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = df[[dependante] + independantes].corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im); ax.set_title("Matrice de corrélation")
        figures.append({"titre": "Matrice de corrélation", "image_base64": _fig_to_base64(fig)})
    except Exception:
        pass
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sm.qqplot(fit.resid, line="s", ax=ax)
        ax.set_title("QQ-plot des résidus (normalité)")
        figures.append({"titre": "QQ-plot des résidus", "image_base64": _fig_to_base64(fig)})
    except Exception:
        pass
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(fit.fittedvalues, fit.resid, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Valeurs prédites"); ax.set_ylabel("Résidus"); ax.set_title("Résidus vs valeurs prédites")
        figures.append({"titre": "Résidus vs prédictions", "image_base64": _fig_to_base64(fig)})
    except Exception:
        pass
    return figures


@app.route("/exporter_xlsx", methods=["POST"])
def exporter_xlsx():
    """Exporte la base nettoyée (celle vraiment utilisée pour l'estimation)
    en fichier Excel téléchargeable, avec un second onglet listant les
    sources/liens d'origine de chaque variable — pour que l'étudiant
    puisse vérifier lui-même qu'aucune donnée n'est inventée."""
    try:
        p = request.get_json(force=True)
        df = pd.DataFrame(p.get("data", []))
        sources = p.get("sources", [])  # [{variable, source, lien}]
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Base nettoyée", index=False)
            if sources:
                pd.DataFrame(sources).to_excel(writer, sheet_name="Sources", index=False)
        stream.seek(0)
        return send_file(stream, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name="base_nettoyee_pain.xlsx")
    except Exception as e:
        return jsonify_propre({"erreur": str(e)}, 500)


@app.route("/variable_dummy", methods=["POST"])
def variable_dummy():
    """Transforme une variable continue trop trouée en variable binaire
    (dummy) par rapport à sa médiane sur les observations disponibles —
    stratégie de repli scientifiquement défendable quand la variable
    d'intérêt a trop de données manquantes pour rester continue."""
    try:
        p = request.get_json(force=True)
        df = pd.DataFrame(p.get("data", []))
        col = p.get("colonne")
        serie = pd.to_numeric(df[col], errors="coerce")
        mediane = float(serie.median())
        df[col] = (serie >= mediane).astype("Int64")
        return jsonify_propre({"data": df.to_dict("records"), "mediane_utilisee": mediane, "note": f"'{col}' transformée en 0/1 selon le seuil médian {mediane} (1 = au-dessus ou égal à la médiane)."})
    except Exception as e:
        return jsonify_propre({"erreur": str(e)}, 500)


@app.route("/rapport", methods=["POST"])
def rapport():
    """Génère un .docx téléchargeable à partir d'un résultat d'estimation
    (reprend l'idée DeepSeek). Le site appelle /estimer d'abord, puis envoie
    ce résultat tel quel ici pour obtenir le document final."""
    try:
        from docx import Document
        from docx.shared import Inches
        p = request.get_json(force=True)
        resultat = p.get("resultat", {})
        hypothese = p.get("hypothese", "")

        doc = Document()
        doc.add_heading("Rapport d'analyse économétrique", 0)
        doc.add_paragraph("Généré par Pain — Delta Infinity")
        doc.add_heading("Hypothèse", level=1)
        doc.add_paragraph(hypothese or "Non précisée")
        doc.add_heading(f"Méthode : {resultat.get('methode', '')}", level=1)

        table = doc.add_table(rows=1, cols=4)
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "Variable", "Coefficient", "Erreur standard", "p-value"
        for c in resultat.get("coefficients", []):
            row = table.add_row().cells
            row[0].text = str(c.get("variable", ""))
            row[1].text = str(c.get("coefficient", ""))
            row[2].text = str(c.get("erreur_standard", ""))
            row[3].text = str(c.get("p_value", ""))

        if resultat.get("r2") is not None:
            doc.add_paragraph(f"R² = {resultat.get('r2')}, n = {resultat.get('n')}")

        for fig in resultat.get("figures", []):
            img_stream = io.BytesIO(base64.b64decode(fig["image_base64"]))
            doc.add_picture(img_stream, width=Inches(5))
            doc.add_paragraph(f"Figure : {fig['titre']}")

        stream = io.BytesIO(); doc.save(stream); stream.seek(0)
        return send_file(stream, mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document", as_attachment=True, download_name="rapport_pain.docx")
    except Exception as e:
        return jsonify_propre({"erreur": str(e)}, 500)


@app.route("/videos", methods=["GET"])
def videos():
    """Suggestions de vidéos pédagogiques (reprend l'idée DeepSeek) — utile
    pour le module de formation. Nécessite une clé YouTube gratuite,
    optionnelle : renvoie une liste vide si absente plutôt que d'échouer."""
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        return jsonify_propre({"videos": [], "note": "YOUTUBE_API_KEY non configurée sur Render — fonctionnalité désactivée, pas bloquante."})
    import requests
    q = request.args.get("q", "")
    try:
        r = requests.get("https://www.googleapis.com/youtube/v3/search", params={"part": "snippet", "q": f"{q} économétrie cours", "type": "video", "maxResults": 5, "key": key}, timeout=8)
        items = r.json().get("items", [])
        return jsonify_propre({"videos": [{"titre": i["snippet"]["title"], "url": f"https://www.youtube.com/watch?v={i['id']['videoId']}", "vignette": i["snippet"]["thumbnails"]["default"]["url"]} for i in items]})
    except Exception as e:
        return jsonify_propre({"videos": [], "erreur": str(e)})


@app.route("/estimer", methods=["POST"])
def estimer():
    try:
        p = request.get_json(force=True)
        methode = p.get("methode")
        dependante = p.get("dependante")
        independantes = p.get("independantes", [])
        data = p.get("data", [])

        if methode in ("thematique", "contenu"):
            return jsonify_propre({"erreur": f"'{methode}' est une analyse qualitative (codage de thèmes/texte), pas un calcul statistique — reste gérée par Pain (Claude) directement, pas par ce moteur."}, 400)
        if methode == "plssem":
            return jsonify_propre({"erreur": "PLS-SEM n'a pas encore de bibliothèque Python fiable/mature intégrée ici (écosystème R plus complet sur ce point). À traiter séparément plus tard."}, 400)

        if methode in ("mco", "mcg", "logit", "probit"):
            if not dependante or not independantes or not data:
                return jsonify_propre({"erreur": "Champs requis : methode, dependante, independantes, data"}, 400)
            df = _clean_dataframe(data, [dependante] + independantes)
            if len(df) < len(independantes) + 2:
                return jsonify_propre({"erreur": f"Pas assez d'observations valides ({len(df)})."}, 400)
            if methode == "mco": resultat = _mco(df, dependante, independantes)
            elif methode == "mcg": resultat = _mcg(df, dependante, independantes)
            else: resultat = _logit_probit(df, dependante, independantes, "Logit" if methode == "logit" else "Probit")

        elif methode == "gmm":
            resultat = _gmm(pd.DataFrame(data), dependante, independantes, p.get("instruments", []))

        elif methode == "panel":
            resultat = _panel(pd.DataFrame(data), dependante, independantes, p.get("entite"), p.get("periode"), p.get("effets", "fixe"))

        elif methode == "ardl":
            resultat = _ardl(pd.DataFrame(data), dependante, independantes, p.get("ar_lags"), p.get("dl_lags"))

        elif methode == "sem":
            resultat = _sem(data, p.get("modele_sem"))

        else:
            return jsonify_propre({"erreur": f"Méthode '{methode}' inconnue."}, 400)

        return jsonify_propre(resultat)

    except Exception as e:
        return jsonify_propre({"erreur": str(e)}, 500)


@app.route("/", methods=["GET"])
def health():
    return jsonify_propre({"status": "ok", "service": "Pain — moteur d'estimation", "methodes": ["mco", "mcg", "logit", "probit", "gmm", "panel", "ardl", "sem"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
