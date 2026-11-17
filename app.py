import json, math, os, pickle
import gradio as gr, pandas as pd

art = pickle.load(open("model.pkl", "rb"))
model, FEATURES, MED = art["model"], art["features"], art["medians"]
SCENARIOS, THRESH = art["scenarios"], art["vuln_threshold"]
metrics = json.load(open("metrics.json")) if os.path.exists("metrics.json") else {}

def predict(assets_bn, nco, npl, roa, tier1, reserve_ratio):
    base = dict(MED)
    base.update(log_assets=math.log(assets_bn*1e6), NTLNLSR=nco, NTLNLSR_lag1=nco, NPERFV=npl, NPERFV_lag1=npl,
                ROA=roa, ROA_lag1=roa, RBC1AAJ=tier1, reserve_ratio=reserve_ratio/100,
                NTLNLSR_chg4=0, NPERFV_chg4=0, ROA_chg4=0)
    rows = []
    for name, macro in SCENARIOS.items():
        x = dict(base); x.update(macro)
        p = float(model.predict(pd.DataFrame([x])[FEATURES])[0])
        rows.append({"Scénario": name, "Pertes prévues t+1 (%)": round(p,3), "Vulnérable": "OUI" if p > THRESH else "non"})
    return pd.DataFrame(rows)

imp = pd.DataFrame(list(metrics.get("_importance", {}).items())[:10], columns=["Variable","Importance"])

with gr.Blocks(title="Stress Test Bancaire IA") as demo:
    gr.Markdown("# 🏦 Stress testing bancaire par IA\nPrévision des pertes sur crédits au trimestre suivant selon le scénario macro.")
    with gr.Row():
        a = gr.Number(10, label="Actifs (Md$)"); n = gr.Number(0.5, label="Taux de pertes actuel (%)"); p = gr.Number(1.0, label="Actifs non performants / actifs (%)")
    with gr.Row():
        r = gr.Number(1.0, label="ROA (%)"); t = gr.Number(12.0, label="Ratio Tier 1 (%)"); rr = gr.Number(1.5, label="Provisions / prêts (%)")
    out = gr.Dataframe(label="Résultats par scénario")
    gr.Button("Lancer le stress test").click(predict, [a,n,p,r,t,rr], out)
    if len(imp):
        gr.Markdown("### Facteurs les plus influents"); gr.Dataframe(imp)
    gr.Markdown("Outil pédagogique, ne constitue pas une évaluation réglementaire.")
demo.launch()
