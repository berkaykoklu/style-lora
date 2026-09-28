"""A blind rating form, as one file with nothing behind it.

The friend rating these is not being paid and should not have to install, sign
up or sign in. So: one HTML file, every image embedded in it, answers kept in
the browser as they go, and a button that hands back a results file at the end.

Blind in the only way that matters here. The items are shuffled and carry no
label, and the key naming which image came from which adapter is written to a
separate file that the form never sees. Someone who knew which was which would
be rating the label.
"""

from __future__ import annotations

import base64
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Small enough that a hundred of them fit in a file worth emailing, large enough
# to judge a brushstroke. 384px JPEG at 80 is about 35 KB.
THUMB = 384
QUALITY = 80

# Three, and no more. The person answering is doing a favour, and every extra
# question multiplies by the number of images.
#
# The first two are the style judgement, asked as two independent scales rather
# than a forced choice so that "neither" is sayable. The third is the one CLIP
# may be too blunt for: its prompt score says the adapter costs nothing in
# content at any strength, which is either true or a limit of the measure, and a
# person looking at a knight can tell which.
QUESTIONS = (
    ("ukiyo", "Ne kadar Japon ağaç baskısı (ukiyo-e)?"),
    ("baroque", "Ne kadar Barok yağlıboya?"),
    ("content", "Yazan şey görselde ne kadar var?"),
)


@dataclass(frozen=True)
class Item:
    path: Path
    source: str  # "base", or the style whose adapter drew it
    prompt: str


def _embed(path: Path) -> str:
    with Image.open(path) as image:
        square = image.convert("RGB").resize((THUMB, THUMB), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    square.save(buffer, "JPEG", quality=QUALITY)
    return base64.b64encode(buffer.getvalue()).decode()


def build(items: list[Item], out_html: Path, out_key: Path, seed: int = 0) -> Path:
    """Write the form and, beside it, the answer key it does not contain."""
    if not items:
        raise ValueError("no items to rate")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    out_key.write_text(
        json.dumps(
            [
                {"n": i + 1, "source": it.source, "prompt": it.prompt}
                for i, it in enumerate(shuffled)
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    cards = []
    for i, item in enumerate(shuffled, start=1):
        rows = "".join(
            f'<div class="q"><p>{text}</p><div class="scale" data-item="{i}" data-q="{key}">'
            + "".join(
                f'<button type="button" data-v="{v}" aria-label="{v}">{v}</button>'
                for v in range(1, 6)
            )
            + "</div></div>"
            for key, text in QUESTIONS
        )
        cards.append(
            f'<article id="card{i}"><header><span class="n">{i} / {len(shuffled)}</span>'
            f'<span class="p">{item.prompt}</span></header>'
            f'<img src="data:image/jpeg;base64,{_embed(item.path)}" alt="">{rows}</article>'
        )

    html = _PAGE.replace("__CARDS__", "\n".join(cards)).replace("__TOTAL__", str(len(shuffled)))
    out_html.write_text(html, encoding="utf-8")
    return out_html


_PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stil değerlendirme</title>
<style>
  :root { --bg:#faf9f7; --card:#fff; --ink:#1b1b1a; --muted:#6b6a67;
          --line:#e3e1dd; --pick:#1b1b1a; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --bg:#141414; --card:#1d1d1c; --ink:#f2f1ee;
      --muted:#9a9894; --line:#302f2d; --pick:#f2f1ee; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:560px; margin:0 auto; padding:16px; }
  h1 { font-size:20px; margin:.6em 0 .2em; }
  .intro { color:var(--muted); font-size:14px; margin-bottom:20px; }
  .bar { position:sticky; top:0; z-index:5; background:var(--bg); padding:10px 0 12px;
         border-bottom:1px solid var(--line); }
  .bar .track { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .bar .fill { height:100%; width:0; background:var(--pick); transition:width .2s; }
  .bar p { margin:8px 0 0; font-size:13px; color:var(--muted); }
  article { background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:14px; margin:16px 0; }
  header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
  .n { font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .p { font-size:13px; color:var(--muted); font-style:italic; text-align:right;
       margin-left:12px; }
  header { gap:8px; }
  img { width:100%; border-radius:8px; display:block; background:var(--line); }
  .q { margin-top:14px; }
  .q p { margin:0 0 8px; font-size:14px; }
  .scale { display:grid; grid-template-columns:repeat(5,1fr); gap:6px; }
  .scale button { padding:12px 0; font:inherit; font-size:15px; cursor:pointer;
                  background:transparent; color:var(--ink);
                  border:1px solid var(--line); border-radius:8px; }
  .scale button[aria-pressed="true"] { background:var(--pick); color:var(--bg);
                                        border-color:var(--pick); }
  .done { position:sticky; bottom:0; background:var(--bg); padding:14px 0 20px;
          border-top:1px solid var(--line); }
  .done button { width:100%; padding:14px; font:inherit; font-size:16px; border-radius:10px;
                 border:1px solid var(--pick); background:var(--pick); color:var(--bg);
                 cursor:pointer; }
  .done button[disabled] { opacity:.4; cursor:not-allowed; }
  .done p { margin:10px 0 0; font-size:13px; color:var(--muted); text-align:center; }
</style></head>
<body><div class="wrap">
<h1>Stil değerlendirme</h1>
<p class="intro">Her görsel için iki soru, 1 (hiç) – 5 (tamamen). Görsellerin nereden
geldiği yazmıyor; ilk izlenimine göre ver, uzun düşünme. Cevapların tarayıcıda
saklanıyor, sekmeyi kapatsan da kaybolmuyor. Sonunda düğmeye basıp çıkan dosyayı
geri yolla.</p>

<div class="bar"><div class="track"><div class="fill" id="fill"></div></div>
<p id="count">0 / __TOTAL__ tamamlandı</p></div>

__CARDS__

<div class="done">
  <button id="save" disabled>Sonuçları indir</button>
  <p id="hint">Hepsi doldurulunca aktifleşir.</p>
</div>
</div>
<script>
  const TOTAL = __TOTAL__, KEY = "style-eval-v1";
  let answers = {};
  try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }

  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
  }
  function complete() {
    let done = 0;
    for (let i = 1; i <= TOTAL; i++) {
      const a = answers[i];
      if (a && a.ukiyo && a.baroque && a.content) done++;
    }
    return done;
  }
  function refresh() {
    const done = complete();
    document.getElementById("fill").style.width = (100 * done / TOTAL) + "%";
    document.getElementById("count").textContent = done + " / " + TOTAL + " tamamlandı";
    const ready = done === TOTAL;
    document.getElementById("save").disabled = !ready;
    document.getElementById("hint").textContent = ready
      ? "Teşekkürler. Dosyayı indirip geri yolla." : "Hepsi doldurulunca aktifleşir.";
  }

  document.querySelectorAll(".scale").forEach(scale => {
    const item = scale.dataset.item, q = scale.dataset.q;
    scale.querySelectorAll("button").forEach(btn => {
      if (answers[item] && String(answers[item][q]) === btn.dataset.v) {
        btn.setAttribute("aria-pressed", "true");
      }
      btn.addEventListener("click", () => {
        scale.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", "false"));
        btn.setAttribute("aria-pressed", "true");
        answers[item] = answers[item] || {};
        answers[item][q] = Number(btn.dataset.v);
        save(); refresh();
      });
    });
  });

  document.getElementById("save").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(answers, null, 2)], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "stil-degerlendirme.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  refresh();
</script></body></html>
"""
