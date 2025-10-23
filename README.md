<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simulated Federated Learning on NSL-KDD using GAT — README</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0b0f14;--panel:#0f1620;--text:#e7eef7;--muted:#9fb0c4;
      --border:#1f2a37;--shadow:0 10px 30px rgba(0,0,0,.35);
      --radius:16px;--accent:#60a5fa;--ok:#34d399;--warn:#fbbf24;--brand:#7dd3fc;
    }
    @media (prefers-color-scheme: light){
      :root{--bg:#f8fafc;--panel:#ffffff;--text:#0f1720;--muted:#4b5563;--border:#e5e7eb;--shadow:0 8px 24px rgba(0,0,0,.08)}
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Arial,Helvetica,sans-serif;line-height:1.7}
    .wrap{max-width:900px;margin:40px auto;padding:0 20px}
    .card{background:linear-gradient(180deg,rgba(255,255,255,.02),transparent);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:22px}
    h1{margin:0 0 8px;font-weight:800;font-size:clamp(26px,3.8vw,38px)}
    p.lead{color:var(--muted);margin:0 0 8px}
    .figs{display:grid;grid-template-columns:1fr;gap:16px;margin:16px 0}
    @media(min-width:860px){.figs{grid-template-columns:1fr 1fr}}
    img{width:100%;height:auto;border-radius:12px;border:1px solid var(--border);box-shadow:var(--shadow)}
    .caption{font-size:12px;color:var(--muted);margin-top:6px}
    table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-top:6px}
    th,td{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left}
    thead th{background:rgba(255,255,255,.04);font-weight:600}
    tbody tr:last-child td{border-bottom:0}
    .pill{display:inline-block;padding:2px 8px;border:1px solid var(--border);border-radius:999px;font-size:12px;color:var(--muted);margin-left:8px}
    .ok{color:var(--ok);font-weight:700}
    .warn{color:var(--warn);font-weight:700}
    .brand{color:var(--brand);font-weight:700}
    footer{color:var(--muted);font-size:12px;margin-top:18px}
    .grid2{display:grid;grid-template-columns:1fr;gap:16px}
    @media(min-width:860px){.grid2{grid-template-columns:1fr 1fr}}
    .kpi{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
    .kpi .pill{padding:6px 10px}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="card">
      <h1>Simulated Federated Learning on NSL-KDD using GAT</h1>
      <p class="lead">Graph Attention Networks trained in a federated setting for intrusion detection on NSL-KDD. Includes privacy-loss vs rounds, loss/accuracy curves, and a t‑SNE view of GAT embeddings.</p>
      <div class="kpi">
        <span class="pill"><strong>Accuracy:</strong> 0.8795</span>
        <span class="pill"><strong>Precision:</strong> 0.8898</span>
        <span class="pill"><strong>Recall:</strong> 0.8552</span>
        <span class="pill"><strong>F1:</strong> 0.8722</span>
        <span class="pill"><strong>AUC:</strong> 0.9139</span>
      </div>
    </header>

    <section class="card" style="margin-top:18px">
      <h2 style="margin:0 0 10px">What this does <span class="pill">simple explanation</span></h2>
      <p>
        We simulate <strong>Federated Learning (FL)</strong> across multiple clients that each hold a shard of NSL‑KDD. 
        A server aggregates client model updates after every communication round (e.g., FedAvg). The model is a 
        <strong>Graph Attention Network (GAT)</strong> built over a graph of samples (e.g., k‑NN graph in feature space). 
        Attention lets the model weigh neighbors when passing messages, which helps capture non‑linear relations between 
        similar traffic records.
      </p>
      <p>
        <strong>Privacy loss vs rounds:</strong> when using differential privacy (DP), the privacy budget (ε) 
        <em>accumulates</em> as rounds increase. More rounds → more exposures of (noised) updates → higher overall ε, 
        meaning weaker privacy guarantees. The plot below illustrates this monotonic increase with training rounds.
      </p>
    </section>

    <section class="card" style="margin-top:18px">
      <h2 style="margin:0 0 10px">Plots</h2>
      <div class="figs">
        <figure>
          <img src="assets/privacy_loss.png" alt="Privacy loss (epsilon) vs federated rounds">
          <figcaption class="caption">Privacy loss (ε) increases with the number of federated rounds (DP composition).</figcaption>
        </figure>
        <figure>
          <img src="assets/loss_accuracy.png" alt="Training loss and accuracy over rounds/epochs">
          <figcaption class="caption">Training dynamics: loss decreases while accuracy improves over time.</figcaption>
        </figure>
      </div>
      <div class="figs" style="margin-top:6px">
        <figure>
          <img src="assets/tsne_gat.png" alt="t-SNE of GAT embeddings">
          <figcaption class="caption">t‑SNE projection of final‑round GAT embeddings; clusters reflect attack vs normal and sub‑types.</figcaption>
        </figure>
      </div>
    </section>

    <section class="card" style="margin-top:18px">
      <h2 style="margin:0 0 10px">Overall metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>AUC</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Federated GAT</td>
            <td class="brand">0.8795</td>
            <td>0.8898</td>
            <td>0.8552</td>
            <td>0.8722</td>
            <td>0.9139</td>
          </tr>
        </tbody>
      </table>
      <footer>Numbers are overall (macro/weighted as configured in code), reported on the held‑out test split.</footer>
    </section>
  </div>
</body>
</html>
