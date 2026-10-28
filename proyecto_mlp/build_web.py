"""Genera la interfaz web. static/index.html llama a /predict (Flask);
demo_web.html es autonoma (inferencia MLP en el navegador con los pesos exportados)."""
import json
m = json.load(open("modelo_web.json"))
r = lambda a: [r(x) for x in a] if isinstance(a, list) else (round(a, 5) if isinstance(a, float) else a)
m = {k: r(v) for k, v in m.items()}
met = json.load(open("metricas.json"))
cm = met["matriz_confusion"]; rep = met["reporte"]

TPL = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Predictor de aprobación · MLP</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap" rel="stylesheet">
<style>
:root{--bg:#F3F6FA;--surf:#fff;--ink:#14213D;--mut:#5A6578;--line:#D8DFEA;--ok:#17805A;--bad:#C93C34;--acc:#2F5BEA;
box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0E1524;--surf:#16203A;--ink:#E8EDF7;--mut:#9AA6BD;--line:#2A3757;--ok:#3FC08F;--bad:#F06B63;--acc:#7C9BFF}}
:root[data-theme="dark"]{--bg:#0E1524;--surf:#16203A;--ink:#E8EDF7;--mut:#9AA6BD;--line:#2A3757;--ok:#3FC08F;--bad:#F06B63;--acc:#7C9BFF}
*{box-sizing:border-box}html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.55 "IBM Plex Sans",system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:28px 18px 56px}
h1{font:600 clamp(26px,4.5vw,38px)/1.15 "IBM Plex Serif",Georgia,serif;margin:0 0 8px;letter-spacing:-.01em}
.sub{color:var(--mut);max-width:62ch;margin:0 0 24px}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surf);border:1px solid var(--line);border-radius:10px;padding:20px}
.f{margin-bottom:16px}.f label{display:flex;justify-content:space-between;font-weight:500;font-size:15px}
.f output{font-variant-numeric:tabular-nums;color:var(--acc);font-weight:600}
.hint{color:var(--mut);font-size:13px;margin:0 0 4px}
input[type=range]{width:100%;accent-color:var(--acc);margin:4px 0}
.sw{display:flex;align-items:center;gap:10px;font-weight:500}
.sw input{width:20px;height:20px;accent-color:var(--acc)}
button{font:600 16px "IBM Plex Sans",sans-serif;background:var(--acc);color:#fff;border:0;border-radius:6px;padding:12px 18px;width:100%;cursor:pointer}
button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.res{text-align:center;padding:10px 0}
.verdict{font:600 34px "IBM Plex Serif",Georgia,serif;margin:6px 0}
.verdict.ok{color:var(--ok)}.verdict.bad{color:var(--bad)}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden;margin:14px 0 6px;position:relative}
.bar i{display:block;height:100%;background:var(--ok);width:0;transition:width .35s}
.bar b{position:absolute;left:50%;top:-2px;bottom:-2px;width:2px;background:var(--ink)}
.small{color:var(--mut);font-size:14px}
h2{font:600 19px "IBM Plex Sans";margin:0 0 10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:center;border:1px solid var(--line)}
td.hit{background:color-mix(in srgb,var(--ok) 18%,transparent);font-weight:600}
td.miss{background:color-mix(in srgb,var(--bad) 14%,transparent)}
.wrap{overflow-x:auto}
details{margin-top:18px}summary{cursor:pointer;font-weight:600}
</style></head><body><main>
<h1>¿Aprobará el estudiante?</h1>
<p class="sub">Un perceptrón multicapa (64 → 32 neuronas ocultas) estima la probabilidad de aprobar a partir de las notas y los hábitos del estudiante. Mueve los controles y pulsa Predecir.</p>
<div class="grid">
<section class="panel" aria-label="Datos del estudiante"><div id="fields"></div>
<div class="f"><label class="sw"><input type="checkbox" id="Extracurriculares"> Participa en talleres o deportes</label></div>
<button id="go" type="button">Predecir</button></section>
<section class="panel" aria-live="polite"><div class="res"><div class="small" id="lead">Resultado</div>
<div class="verdict" id="v">—</div>
<div class="bar"><i id="pb"></i><b></b></div>
<div class="small" id="pt">Pulsa Predecir para ver la probabilidad de aprobar.</div></div>
<details open><summary>Desempeño en el set de prueba (120 estudiantes)</summary>
<div class="wrap" style="margin-top:10px"><table><tr><th></th><th>Pred. Reprobado</th><th>Pred. Aprobado</th></tr>
<tr><th>Real Reprobado</th><td class="hit">__C00__</td><td class="miss">__C01__</td></tr>
<tr><th>Real Aprobado</th><td class="miss">__C10__</td><td class="hit">__C11__</td></tr></table></div>
<p class="small">Accuracy __ACC__ · Precisión __PRE__ · Recall __REC__ · F1 __F1__ (clase Aprobado)</p></details></section></div>
</main>
<script>
const MODEL=__MODEL__;
const META=[["Tareas","Tareas (0–100)",0,100,1,65],["Examenes","Exámenes (0–100)",0,100,1,60],["Asistencia","Asistencia % (0–100)",0,100,1,70],["Estudio_Horas","Horas de estudio por semana",0,25,0.5,13],["Sueno_Horas","Horas de sueño por día",3,10,0.5,6.5]];
const box=document.getElementById("fields");
META.forEach(([k,l,a,b,s,d])=>{box.insertAdjacentHTML("beforeend",`<div class="f"><label for="${k}">${l}<output id="o_${k}">${d}</output></label><input type="range" id="${k}" min="${a}" max="${b}" step="${s}" value="${d}"></div>`);
document.getElementById(k).addEventListener("input",e=>document.getElementById("o_"+k).textContent=e.target.value)});
function relu(x){return x>0?x:0}
function sigmoid(z){return 1/(1+Math.exp(-z))}
function local(row){ // imputar mediana -> estandarizar -> capas ocultas ReLU -> sigmoide
  let a=MODEL.features.map((f,i)=>{let v=row[f];if(v===null||isNaN(v))v=MODEL.medianas[i];return (v-MODEL.mean[i])/MODEL.scale[i]});
  const L=MODEL.W.length;
  for(let l=0;l<L;l++){const W=MODEL.W[l],b=MODEL.b[l];const o=b.slice();
    for(let j=0;j<o.length;j++){for(let i=0;i<a.length;i++)o[j]+=a[i]*W[i][j]}
    a=l<L-1?o.map(relu):o.map(sigmoid)}
  return a[0]}
async function predict(row){__PREDICT__}
document.getElementById("go").addEventListener("click",async()=>{
  const row={};META.forEach(([k])=>row[k]=parseFloat(document.getElementById(k).value));
  row.Extracurriculares=document.getElementById("Extracurriculares").checked?1:0;
  const v=document.getElementById("v"),pt=document.getElementById("pt");
  try{const p=await predict(row);const ok=p>=0.5;
    v.textContent=ok?"Aprobado":"Reprobado";v.className="verdict "+(ok?"ok":"bad");
    document.getElementById("pb").style.width=(p*100).toFixed(1)+"%";
    pt.textContent="Probabilidad de aprobar: "+(p*100).toFixed(1)+" %. La línea central marca el umbral de 50 %."}
  catch(e){v.textContent="Error";v.className="verdict bad";pt.textContent="No se pudo calcular la predicción: "+e.message}});
</script></body></html>"""

def build(predict_js, out, embed=True):
    s = TPL.replace("__PREDICT__", predict_js)
    s = s.replace("__MODEL__", json.dumps(m) if embed else "null")
    a = rep["Aprobado"]
    for k, v in {"__C00__": cm[0][0], "__C01__": cm[0][1], "__C10__": cm[1][0], "__C11__": cm[1][1],
                 "__ACC__": f"{met['accuracy_test']*100:.1f} %", "__PRE__": f"{a['precision']*100:.1f} %",
                 "__REC__": f"{a['recall']*100:.1f} %", "__F1__": f"{a['f1-score']*100:.1f} %"}.items():
        s = s.replace(k, str(v))
    open(out, "w", encoding="utf-8").write(s)

# Version servicio: llama al endpoint REST /predict
build("""const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(row)});
const d=await r.json();if(!r.ok)throw new Error((d.error||[]).join(", "));return d.probabilidad_aprobar;""",
      "static/index.html", embed=False)
# Version demo autonoma (inferencia local con los pesos del MLP)
build("return local(row);", "/mnt/user-data/outputs/demo_web.html", embed=True)
