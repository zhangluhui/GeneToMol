"""GeneToMol -- Streamlit front end. Gene expression in, molecule out.

    streamlit run app.py

Reads a bundle built by `genetomol.export_serving` and never imports torch.
The readers and quality guards are imported from `genetomol.retrieve` rather
than reimplemented, so the web path and the CLI cannot drift apart.

Layout is quality first, hits second: gene coverage and replicate reliability
render above the ranked table, and a query that fails the guards produces no
table at all.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from genetomol.retrieve import (
    MIN_COVERAGE,
    _read,
    describe_reliability,
    gene_aliases,
    per_gene_z,
    robust_z,
    split_half_reliability,
)
from genetomol.serve import Bundle

ROOT = Path(__file__).parent
BUNDLE = ROOT / "serving_bundle.npz"
# Optional, and absent in a normal deployment: the bundle already carries the
# landmark aliases. Point GENETOMOL_GENEINFO at a local `geneinfo_beta.txt` only
# to widen the mapping beyond the 978 landmarks.
# Path("") is Path("."), which exists and is truthy -- so the empty default
# has to become None here, or an unset variable points the reader at the
# working directory.
_geneinfo_env = os.environ.get("GENETOMOL_GENEINFO", "").strip()
GENEINFO = Path(_geneinfo_env) if _geneinfo_env else None
EXAMPLES = ROOT / "examples"
ARCH = ROOT / "docs" / "model_structure.png"
PERF = ROOT / "docs" / "performance.png"

st.set_page_config(
    page_title="GeneToMol", page_icon="🧪", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/zhangluhui/GeneToMol",
        "Report a bug": "https://github.com/zhangluhui/GeneToMol/issues",
        "About": "**GeneToMol** \u2014 AI-driven compound identification from "
                 "transcriptomic signatures. Cosine is a similarity, not a "
                 "probability: read the ranked list as a shortlist to test.",
    })

# Streamlit leaves ~6rem above the first element, which pushes the title below
# the fold on a laptop. Everything here is spacing only.
st.markdown("""
<style>
  .block-container {padding-top: 2.6rem; padding-bottom: 3rem;}
  section[data-testid="stSidebar"] .block-container {padding-top: 2rem;}
  h1 {letter-spacing: -0.02em;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path: str, mtime: float) -> Bundle:
    """Cached across sessions. `mtime` busts the cache when the bundle changes."""
    return Bundle(path)


@st.cache_resource(show_spinner="Loading gene aliases...")
def load_aliases(path: str) -> dict:
    return gene_aliases(path) if Path(path).is_file() else {}


def read_with_aliases(path, genes, aliases):
    """`_read` also accepts an already-built mapping, which is how the bundle's
    own aliases are used without any LINCS metadata on the serving machine."""
    return _read(str(path), genes, aliases)


def read_upload(upload, genes, aliases):
    """Streamlit hands us bytes; `_read` wants a path-like, since it sniffs the
    separator from the extension. Spool to a per-call temp dir -- never the app
    directory, where two sessions uploading the same filename would collide."""
    with tempfile.TemporaryDirectory() as d:
        q = Path(d) / Path(upload.name).name
        q.write_bytes(upload.getvalue())
        return read_with_aliases(q, genes, aliases)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("GeneToMol")
st.sidebar.caption(
    "AI-driven compound identification from transcriptomic signatures")
st.sidebar.caption(
    "by Luhui Zhang · [GitHub](https://github.com/zhangluhui/GeneToMol)")
if not BUNDLE.exists():
    st.error(
        f"No serving bundle at `{BUNDLE}`.\n\nBuild one with:\n\n"
        "```\npython -m genetomol.export_serving \\\n"
        "  --data ../lincs/prepared --run runs/genetomol \\\n"
        "  --mol-features ../lincs/prepared/ecfp_chiral.npz \\\n"
        "  --out serving_bundle.npz\n```")
    st.stop()

bundle = load_bundle(str(BUNDLE), BUNDLE.stat().st_mtime)
# Aliases ride inside the bundle. A local geneinfo file, if one happens to be
# present, only adds to them -- it is never required.
ALIASES = dict(bundle.aliases)
if GENEINFO is not None and GENEINFO.is_file():
    ALIASES.update(load_aliases(str(GENEINFO)))
aliases_available = bool(ALIASES)

st.sidebar.caption(
    f"**{len(bundle):,}** compounds · **{len(bundle.genes)}** landmark genes")
st.sidebar.divider()

TASKS = {
    "library-matching": (
        "Library matching",
        "Assumes the compound behind your signature is already in the compound library.\n\n"
        "On held-out test data:\n\n"
        "**20.1%** of queries put the exact compound in the top 10, "
        "**35.1%** in the top 50, **51.3%** in the top 200.\n\n"
        "**53.2%** of queries put the compound having same targets in "
        "the top 10, **72.4%** in the top 50, **86.1%** in the top 200.")
}
label, blurb = TASKS.get(bundle.task, ("Unknown task", ""))
st.sidebar.info(f"**{label}**\n\n{blurb}")

k = st.sidebar.slider("Compounds to return", 5, 100, 25, step=5)
return_all = st.sidebar.checkbox(
    f"Return all {len(bundle):,} compounds", value=False,
    help="Rank the entire library instead of the top slice. The table stays "
         "scrollable and the CSV download contains every row.")
if return_all:
    k = len(bundle)

dispersion = st.sidebar.selectbox(
    "Dispersion", ["pooled (recommended)", "per-gene"],
    help="How the treated-minus-control contrast is put on a common scale.\n\n"
         "**pooled** divides every gene by a single number measuring how much "
         "the whole contrast varies. It is stable with only a few replicates, "
         "which is why it is the default.\n\n"
         "**per-gene** gives each gene its own divisor, measured from how much "
         "that gene varies across your control samples. It can describe your "
         "data more faithfully when you have many replicates to estimate those "
         "divisors from; with only a handful the divisors are themselves noisy "
         "and a few genes can blow up.")
take_log = st.sidebar.checkbox(
    "Values are linear (apply log2)", value=True,
    help="Tick for normalized expression on a linear scale -- TPM, CPM, "
         "normalized counts. Untick if the values are already log-scale.")

st.sidebar.divider()
st.sidebar.caption(
    "Cosine is a similarity, not a probability. Read the ranked list as a "
    "shortlist to test.")

# ---------------------------------------------------------------- main
tab_run, tab_model = st.tabs(["Retrieve", "How it works"])

with tab_model:
    st.header("What it is for")
    st.markdown("""
A cell responding to a drug leaves a readable trace. Thousands of genes shift
together, and that pattern is close to a fingerprint of what the compound did
to the cell -- which pathway it hit, which mechanism it engaged. Reading that
fingerprint backwards shows up in three places:

- **Deconvolution.** A phenotypic screen found a hit and nobody knows what it
  is. Match its signature against a library already profiled.
- **Mechanism of action.** A compound works and nobody knows why. The compounds
  it retrieves share a mechanism far more often than chance, so the ranked list
  is a hypothesis about the target.
- **Drug repurposing.** Take the signature of a disease state, invert it, and
  ask which existing compounds push the cell the other way.

The classical answer is the Connectivity Map (CMap): correlate the query
against each compound's average profile and sort. That has a hard ceiling --
correlation compares a signature only with other signatures, so a compound that
was never profiled cannot be scored at all. GeneToMol learns a shared space
instead, and because the molecule side is a *function of structure* rather than
a lookup of past measurements, it can score chemistry it has never seen.
""")

    st.divider()
    st.header("Model structure")
    if ARCH.exists():
        st.image(str(ARCH), width="stretch")
    st.markdown(f"""
Two encoders are trained to put a **transcriptomic signature** and a
**molecule** at the same place in one 256-dimensional space. Retrieval is then
a single matrix multiply: encode the query, take the cosine against all
{len(bundle):,} pre-encoded compounds, sort.

**Signature encoder** -- 978 landmark genes to 256 dims, through
`Linear - LayerNorm - GELU - Dropout`. It sees the expression vector and
nothing else: no cell line, no dose, no timepoint. A real query carries none
of those either, so the model is never handed anything during testing that it
would lack in practice, and the figures below are what you should actually
expect rather than a best case.

**Molecule encoder** -- `2063 -> 1024 -> 512 -> 256`. The input is ECFP4 with
chirality: 2048 fingerprint bits plus 15 physicochemical descriptors. Both
towers end at 256 so their outputs land in the same space. Because it is a
*function of the molecule* rather than a lookup table, it can score a compound
it has never seen.

**Training objective** -- symmetric InfoNCE, the CLIP recipe. Each signature's
own compound is the positive and the other 255 in the batch are negatives,
scaled by a learned temperature.

### How well it does

Everything below is measured on **held-out test** data.

""")
    if PERF.exists():
        st.image(str(PERF), width="stretch")
    st.markdown(f"""

Three things the curves say that a single headline number hides.

**Left -- recall per compound.** Every compound counts once, however often it
was profiled: **20% at k=10, 35% at k=50, 51% at k=200**, so a top-200
shortlist out of 24,000 contains the right molecule half the time.

**Middle -- recall per signature.** Weighted by profiling depth, which reflects
how screening effort was allocated rather than biology. It reads higher --
**39% at k=10** -- because heavily-profiled compounds are both over-represented
and easier.

**Right -- mechanism instead of molecule.** Asking only that *some* top-k hit
shares the query's MOA or target. **54% at k=10 against 20% for the exact
compound**, and the two curves for MOA and target track each other closely.
If a same-mechanism answer is useful to you, that is the number to plan
around -- and it is why the results table shows the MOA column.

### What the numbers mean

| | |
|---|---|
| Candidate bank | {len(bundle):,} compounds |
| Query | {len(bundle.genes)} landmark genes |
| Shared space | 256 dims, cosine |
| Task | **{label}** |

{blurb}

### Honest limits

- **Cosine is a similarity, not a probability.** No calibration has been
  measured, so rank order is meaningful and the absolute value is not.
- **Mechanism is easier than identity.** A same-MOA compound reaches the top 10
  about 54% of the time where the exact compound reaches it 20%. Read agreement
  across the top hits.
- **Query quality dominates.** Retrieval from a perfect signature reaches 74%
  at k=10.
""")

with tab_run:
    st.title("GeneToMol")
    st.markdown("##### Gene expression in, molecule out")
    st.caption(
        "Upload treated and control samples; the model ranks all "
        f"{len(bundle):,} LINCS compounds by cosine similarity in a learned "
        "space. Cosine is a similarity, not a probability.")
    st.divider()

    with st.expander("Input format", expanded=False,
                     icon=":material/table_view:"):
        st.markdown(
            "**Treated** is the state you are asking about -- cells after "
            "exposure to the compound you want to identify. **Control** is "
            "the reference it is compared against: untreated or "
            "vehicle-treated cells. The model reads the difference between "
            "them.\n\n")
        st.markdown(
            "CSV or TSV with a header. **First column = gene identifier**, every "
            "other column = one sample.\n\n"
            "```\ngene_symbol,rep1,rep2\nAARS1,12.3,11.8\nABCB6,4.1,4.6\n```\n\n"
            "Identifiers may be **Entrez ids**, **gene symbols** or **Ensembl ids**"
            + (f" ({len(ALIASES):,} aliases available)." if aliases_available else
               " -- but this bundle carries no aliases, so only Entrez ids map "
               "here. Rebuild it with --geneinfo.") +
            " Row order does not matter; values are reindexed onto the "
            f"{len(bundle.genes)} landmark genes.")

    # --- example ---------------------------------------------------------
    ex_t, ex_c = EXAMPLES / "example_treated.csv", EXAMPLES / "example_control.csv"
    if ex_t.exists() and ex_c.exists():
        with st.expander("No data to hand? Try the worked example",
                         expanded=True, icon=":material/science:"):
            st.markdown(
                "Three treated and three control samples, gene symbols, linear "
                "scale -- the shape a real RNA-seq table arrives in. It was "
                "built by adding a **held-out test** LINCS signature to a "
                "synthetic baseline, so the right answer is known and the "
                "pipeline has to earn it.\n\n"
                "Download both, then upload them below with **Values are "
                "linear** ticked.")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Treated (3 reps)", ex_t.read_bytes(),
                               file_name="example_treated.csv", mime="text/csv")
            d2.download_button("Control (3 reps)", ex_c.read_bytes(),
                               file_name="example_control.csv", mime="text/csv")
            if d3.button("Run it now", type="primary"):
                st.session_state["use_example"] = True
            ans = EXAMPLES / "example_answer.json"
            if ans.exists():
                a = json.loads(ans.read_text())
                st.caption(
                    f"Answer, once you have looked: **{a['compound_id']}** "
                    f"- {a['moa']} (target {a['target']}). Every one of the top "
                    "10 shares that mechanism, which is what MOA agreement "
                    "looks like when it works.")
            st.info(
                "This example is clean and was chosen to be **informative, not "
                "typical**. A random held-out test query puts the exact compound "
                "in the top 10 about 20% of the time.")

    per_gene = dispersion.startswith("per-gene")
    c1, c2 = st.columns(2)
    treated = c1.file_uploader(
        "Treated samples", type=["csv", "tsv", "txt"],
        help="The state you are asking about: cells after exposure to the "
             "compound you want to identify.")
    control = c2.file_uploader(
        "Control samples", type=["csv", "tsv", "txt"],
        help="The reference the treated samples are compared against: "
             "untreated or vehicle-treated cells.")

    use_example = st.session_state.get("use_example") and treated is None
    if not use_example and (treated is None or control is None):
        st.info("Upload both files to begin, or open the worked example above.")
        st.stop()

    aliases_path = ALIASES
    try:
        if use_example:
            # Force the example's own settings so a stale sidebar cannot make
            # the demo look broken.
            take_log = True
            st.success("Running the worked example.")
            T, t_names, t_rep = read_with_aliases(ex_t, bundle.genes, ALIASES)
            C, c_names, _ = read_with_aliases(ex_c, bundle.genes, ALIASES)
        else:
            T, t_names, t_rep = read_upload(treated, bundle.genes, aliases_path)
            C, c_names, _ = read_upload(control, bundle.genes, aliases_path)
    except SystemExit as exc:
        st.error(str(exc))
        st.stop()

    # ------------------------------------------------ build the query + guards
    problems: list[str] = []
    notes: list[str] = []
    rel = None

    if take_log:
        if min(np.nanmin(T), np.nanmin(C)) < 0:
            problems.append(
                "'Values are linear' is ticked but the data contains negative "
                "numbers, so it is already log-scale or already a differential.")
        else:
            T, C = np.log2(T + 1.0), np.log2(C + 1.0)
    keep = ~(np.isnan(T).any(0) | np.isnan(C).any(0))
    cov = float(keep.mean())
    if not problems:
        rel = split_half_reliability(T[:, keep], C[:, keep])
        tm = T[:, keep].mean(0)
        z = np.zeros(len(bundle.genes))
        z[keep] = (per_gene_z(tm, C[:, keep]) if per_gene
                   else robust_z(tm - C[:, keep].mean(0)))
        X = z[None, :]
        names = ["treated - control"]

    if cov < MIN_COVERAGE:
        problems.append(
            f"Only **{int(keep.sum())}/{len(bundle.genes)}** landmark genes "
            f"({cov:.0%}) matched. Below {MIN_COVERAGE:.0%} the zero-fill dominates."
            + ("" if aliases_available else
               " `geneinfo_beta.txt` is missing, so symbols cannot be mapped."))
    elif cov < 0.9:
        notes.append(f"{len(bundle.genes) - int(keep.sum())} landmark genes were "
                     "filled with 0.0 (no change). Hits are less reliable.")

    # ------------------------------------------------------------ quality panel
    st.subheader("Query quality")
    m1, m2, m3 = st.columns(3)
    m1.metric("Landmark genes matched", f"{int(keep.sum())}/{len(bundle.genes)}",
              f"{cov:.0%}", border=True)
    m2.metric("Replicates", f"{T.shape[0]} treated / {C.shape[0]} control",
              border=True)
    if rel:
        sb = rel["averaged_r_spearman_brown"]
        m3.metric("Replicate reliability (rho)", f"{sb:+.3f}",
                  delta=("good" if sb >= 0.40 else "thin" if sb >= 0.15 else "poor"),
                  delta_color=("normal" if sb >= 0.40 else
                               "off" if sb >= 0.15 else "inverse"), border=True)
    else:
        m3.metric("Replicate reliability", "n/a", delta="need 2+ per side",
                  delta_color="off", border=True)

    if problems:
        for p in problems:
            st.error(p)
        st.stop()
    for n in notes:
        st.warning(n)

    if rel:
        sb = rel["averaged_r_spearman_brown"]
        msg = (f"Split-half reliability **{sb:+.3f}** "
               f"(single contrast {rel['single_contrast_r']:+.3f}, "
               f"{rel['n_pairings']} disjoint pairings). {describe_reliability(sb)}")
        (st.success if sb >= 0.40 else st.warning if sb >= 0.15 else st.error)(msg)
        if sb < 0.15:
            st.error(
                "**These results are unlikely to be meaningful.** More replicates "
                "help more than anything else: averaging k replicates takes "
                "reliability r to k·r/(1+(k−1)·r).")
    else:
        st.warning(
            "One replicate per side: no reliability estimate is possible, and no "
            "replicate averaging was done. LINCS consensus signatures score 0.449; "
            "an unreplicated contrast sits below the shallowest bin measured (0.168).")

    if per_gene:
        st.warning(
            f"**per-gene dispersion**, scaling each gene by its own median "
            f"absolute deviation across your **{C.shape[0]}** control sample(s). "
            "With few controls those divisors are themselves noisy and a handful "
            "of genes can dominate the query; the more controls you have, the "
            "more trustworthy this becomes. Compare against **pooled** before "
            "settling on either.")

    # A contrast is normally well-scaled, but per-gene dispersion can blow it up
    # by two orders of magnitude when a gene's controls happen to agree, and
    # that is silent otherwise.
    sd = float(np.std(X[:, keep]))
    if not 0.2 < sd < 12:
        st.error(
            f"Query spread is **sd={sd:.2f}**. LINCS z-scores are typically "
            "sd 1-3, so this is far outside the range the model was trained on "
            "and the ranking is unlikely to mean anything."
            + (" This is the signature of per-gene dispersion dividing by a "
               "near-zero deviation; switch **Dispersion** back to **pooled**."
               if per_gene else ""))

    # ------------------------------------------------------------------ hits
    st.subheader("Ranked compounds")
    all_hits = bundle.hits(X, k)

    # The app keeps telling people to read MOA agreement rather than rank 1, so
    # compute it for them instead of leaving it to be eyeballed.
    top = all_hits[0]
    head = top[:10]
    moas = [h["moa"] for h in head if h["moa"] and h["moa"] != "-"]
    top_moa, n_moa = ("", 0)
    if moas:
        top_moa = max(set(moas), key=moas.count)
        n_moa = moas.count(top_moa)
    s1, s2, s3 = st.columns(3)
    s1.metric("Top hit", top[0]["compound_id"], border=True)
    s2.metric("Best cosine", f"{top[0]['cosine']:.3f}", border=True)
    s3.metric("MOA agreement in top 10",
              f"{n_moa}/10" if n_moa else "no annotations",
              delta=(top_moa[:34] if n_moa else None), delta_color="off",
              border=True)
    st.caption(f"Showing {len(top):,} of {len(bundle):,} compounds.")
    for name, rows in zip(names, all_hits):
        if len(names) > 1:
            st.markdown(f"**{name}**")
        st.dataframe(
            rows, width="stretch", hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                # Deliberately a number, not a progress bar: a bar reads as a
                # proportion of something, and cosine is not calibrated.
                "cosine": st.column_config.NumberColumn(format="%.4f",
                                                        width="small"),
                "compound_id": st.column_config.TextColumn("compound",
                                                           width="medium"),
                "moa": st.column_config.TextColumn("MOA", width="large"),
                "target": st.column_config.TextColumn("target", width="small"),
                "smiles": st.column_config.TextColumn("SMILES", width="medium"),
            })

    flat = [dict(query=n, **h) for n, rows in zip(names, all_hits) for h in rows]
    import pandas as pd

    st.download_button("Download hits (CSV)",
                       pd.DataFrame(flat).to_csv(index=False).encode(),
                       file_name="compound_hits.csv", mime="text/csv",
                       icon=":material/download:")

    st.caption(
        "Cosine is a similarity, not a probability, and no calibration has been "
        "measured. Look for **MOA agreement across the top hits** rather than "
        "trusting rank 1: on held-out test data a same-mechanism compound reaches "
        "the top 10 about 54% of the time where the exact compound reaches it 20%.")

    st.divider()
    st.caption(
        f"GeneToMol · {len(bundle):,} compounds · {len(bundle.genes)} landmark "
        "genes · see **How it works** for what the numbers mean")
