# app.py
import io
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

# -----------------------------
# Page config & simple theme
# -----------------------------
st.set_page_config(page_title="Openness Assessment", layout="wide")
PRIMARY = "#1F4E5F"   # petroleum blue
ACCENT  = "#3BAA8F"   # aqua green
LIGHT   = "#F4F6F8"   # light gray

CUSTOM_CSS = """
<style>
:root{
  --primary: #1F4E5F;
  --bg-card: #FFFFFF;
  --fg-card: #111827;
  --border:  #E6E8EB;
  --muted:   #6B7280;
  --section: #F4F6F8;
}
html[data-theme="dark"]{
  --primary: #9AD0DF;
  --bg-card: #111827;
  --fg-card: #E5E7EB;
  --border:  #334155;
  --muted:   #9CA3AF;
  --section: #0B1220;
}

/* Títulos */
h1, h2, h3 { color: var(--primary); }

/* Tarjetas (Recommendations) */
.card {
  background: var(--bg-card);
  color: var(--fg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  margin-bottom: 0.8rem;
}
.card * { color: inherit; }            /* TODO lo interno hereda el color */
.block-title { 
  font-weight: 700; 
  font-size: 1.1rem; 
  color: var(--primary); 
  margin-bottom: 0.35rem;
}

/* Secciones grises */
.section {
  background: var(--section);
  border-radius: 16px;
  padding: 1rem 1.1rem;
}

/* Links */
a { color: #2f9683; }
html[data-theme="dark"] a { color: #86d0c0; }

/* Acentos de clasificación (bordes izquierdos) */
.card-low    { border-left: 6px solid #D9534F; }
.card-medium { border-left: 6px solid #F0AD4E; }
.card-high   { border-left: 6px solid #5CB85C; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# MODE A) Questionnaire (1–5) with embedded APERTO grid + Radar/Polar Bar
# =============================================================================
APERTO_ITEMS = [
  {"Dimension of openness":"OPEN NETWORK","Variable":"ON1","Questions":"How is the partnership regarding new actors?","Scale_5_text":"The network is able to connect and integrate new actors into the collaboration.","Scale_1_text":"The partnership is concentrated around a fixed number of actors."},
  {"Dimension of openness":"OPEN NETWORK","Variable":"ON2","Questions":"How would you characterize the network of stakeholders?","Scale_5_text":"The stakeholder network is decentralized: no single leader; freedom for action.","Scale_1_text":"The stakeholder network is centralized: a leader is clearly identified and defines actions."},
  {"Dimension of openness":"OPEN NETWORK","Variable":"ON3","Questions":"How would you define the management of the innovation process?","Scale_5_text":"Self-management and self-organization.","Scale_1_text":"Controlled from outside."},
  {"Dimension of openness":"OPEN NETWORK","Variable":"ON4","Questions":"How would you define the governance of the innovation partnership?","Scale_5_text":"Governance is clear, transparent, and effective for all partners.","Scale_1_text":"Governance is unclear with no formalized rules."},

  {"Dimension of openness":"OPEN COMMUNICATION","Variable":"OC1","Questions":"How would you define information sharing in the partnership?","Scale_5_text":"All partners trust shared information; it is reliable and complete.","Scale_1_text":"Information is unreliable and partners cannot validate it."},
  {"Dimension of openness":"OPEN COMMUNICATION","Variable":"OC2","Questions":"How would you define the partnership in terms of communication?","Scale_5_text":"Open communication: all partners freely access the information they need.","Scale_1_text":"Restricted communication: partners have limited access to necessary information."},
  {"Dimension of openness":"OPEN COMMUNICATION","Variable":"OC3","Questions":"How would you define data and knowledge documentation?","Scale_5_text":"Shared documentation that is searchable, reusable, and versioned.","Scale_1_text":"No shared documentation or ad-hoc fragmented notes."},

  {"Dimension of openness":"OPEN DESIGN","Variable":"OD1","Questions":"How would you define co-design practices?","Scale_5_text":"Multi-actor co-design with early and continuous user/partner feedback.","Scale_1_text":"Design decided by a small core team with limited consultation."},
  {"Dimension of openness":"OPEN DESIGN","Variable":"OD2","Questions":"How open are the specifications and artifacts?","Scale_5_text":"Specifications/artifacts are open and reusable across partners.","Scale_1_text":"Specifications/artifacts are closed or only shared case-by-case."},

  {"Dimension of openness":"OPEN SPACE","Variable":"OSp1","Questions":"How would you define working spaces and collaboration settings?","Scale_5_text":"Shared, inclusive, accessible spaces (physical/virtual) that foster collaboration.","Scale_1_text":"Fragmented or private spaces with limited partner access."},
  {"Dimension of openness":"OPEN SPACE","Variable":"OSp2","Questions":"How would you define the availability of tools and infrastructure?","Scale_5_text":"Common toolsets/infrastructure with fair access policies.","Scale_1_text":"Tools/infrastructure are siloed; access is restricted or opaque."},

  {"Dimension of openness":"OPEN USE","Variable":"OU1","Questions":"How would you define usage rights for outputs and resources?","Scale_5_text":"Permissive use by all partners (clear licenses, minimal barriers).","Scale_1_text":"Restrictive use; unclear licenses or heavy gatekeeping."},
  {"Dimension of openness":"OPEN USE","Variable":"OU2","Questions":"How would you define reuse and adaptation of outputs?","Scale_5_text":"Outputs are designed for reuse, adaptation, and scaling.","Scale_1_text":"Outputs are difficult to reuse or adapt."},

  {"Dimension of openness":"OPEN RESEARCH","Variable":"OR1","Questions":"How would you define participation in the research process?","Scale_5_text":"Broad, structured participation of stakeholders across stages.","Scale_1_text":"Narrow participation; decisions concentrated in a few actors."},
  {"Dimension of openness":"OPEN RESEARCH","Variable":"OR2","Questions":"How would you define data practices in research?","Scale_5_text":"Data are FAIR (findable, accessible, interoperable, reusable).","Scale_1_text":"Data are closed, poorly documented, or hard to access."},
  {"Dimension of openness":"OPEN RESEARCH","Variable":"OR3","Questions":"How would you define transparency and reproducibility?","Scale_5_text":"Transparent methods; shared code/materials for reproducibility.","Scale_1_text":"Opaque methods; materials/code unavailable."},

  {"Dimension of openness":"OPEN SOCIETY","Variable":"OS1","Questions":"How would you define societal engagement?","Scale_5_text":"Active engagement with communities; inclusive benefits considered.","Scale_1_text":"Limited or no engagement with affected communities."},
  {"Dimension of openness":"OPEN SOCIETY","Variable":"OS2","Questions":"How would you define alignment with public interest?","Scale_5_text":"Clear alignment; public value and ethics integrated into decisions.","Scale_1_text":"Public interest is not explicitly considered."},

  {"Dimension of openness":"OPEN MIND","Variable":"OM1","Questions":"How would you define attitudes toward collaboration and learning?","Scale_5_text":"High trust, curiosity, reflexivity; willingness to learn and adapt.","Scale_1_text":"Low trust; resistance to change or external collaboration."},
  {"Dimension of openness":"OPEN MIND","Variable":"OM2","Questions":"How would you define the culture of feedback?","Scale_5_text":"Constructive feedback loops are institutionalized.","Scale_1_text":"Feedback is sporadic, defensive, or discouraged."}
]
APERTO_COLS = ["Dimension of openness","Variable","Questions","Scale_5_text","Scale_1_text"]

# =============================================================================
# RECOMMENDATIONS (per dimension & level)
# =============================================================================
RECS = {
    "OPEN NETWORK": {
        "Low":   "Map missing stakeholders beyond recurrent partners; publish open onboarding (MoUs, light-weight entry rules) and rotate facilitation to avoid centralization.",
        "Medium":"Consolidate shared governance: mixed committee (university–industry–gov–community), clear decision logs, and periodic inclusion surveys.",
        "High":  "Replicate your partner-selection methodology in new territories; publish playbook and capture lessons learned per cycle."
    },
    "OPEN RESEARCH": {
        "Low":   "Run applied-research pilots with producers/extensionists before scaling; co-define success metrics and minimum documentation (protocol + data sheet).",
        "Medium":"Formalize transfer capabilities: reproducible notebooks, data dictionaries, and short briefs for non-technical audiences.",
        "High":  "Open a pipeline for spin-offs and shared IP; align with international partners and create a fund for open replication studies."
    },
    "OPEN ACCESS": {
        "Low":   "Translate technical outputs into accessible formats (how-to guides, infographics) and assign an owner for each public artifact.",
        "Medium":"Deploy an open repository with versioning and usage analytics; offer short trainings on data use and citation.",
        "High":  "Institutionalize default-open policies with exceptions and create a public observatory with regular updates."
    },
    "OPEN DESIGN": {
        "Low":   "Make early co-design sessions mandatory with key user groups; add a simple usability checklist to each iteration.",
        "Medium":"Standardize participatory protocols for pilots and validations (who, when, feedback loop, change log).",
        "High":  "Implement continuous multi-actor feedback with scheduled redesign sprints and traceability from insight → spec change."
    },
    "OPEN SOCIETY": {
        "Low":   "Activate systematic engagement with affected communities; include equity criteria in milestones and feedback capture.",
        "Medium":"Anchor participatory governance in local/regional regulation; publish roles, rights and escalation paths.",
        "High":  "Build autonomous territorial ecosystems with institutional and financial sustainability (local funds + anchor orgs)."
    },
    "OPEN MIND": {
        "Low":   "Invest in trust-building and listening spaces; run retrospectives emphasizing learning and shared responsibility.",
        "Medium":"Offer mindset workshops (lab → entrepreneurship); pair researchers with field practitioners for short rotations.",
        "High":  "Consolidate a hybrid culture science–enterprise–community with peer mentoring and recognition for boundary spanners."
    },
    "OPEN SPACE": {
        "Low":   "Enable accessible physical/virtual collaboration spaces; define minimal facilitation and inclusive access rules.",
        "Medium":"Secure shared infrastructure (labs, makerspaces) with booking transparency and governance of maintenance.",
        "High":  "Design multi-functional venues mixing innovation, cultural identity and entrepreneurship with open programming."
    },
    "OPEN COMMUNICATION": {
        "Low":   "Create basic feedback channels and a single source of truth; define information owners and validation steps.",
        "Medium":"Guarantee traceability of decisions (minutes + rationale); maintain a searchable knowledge base with versioning.",
        "High":  "Operate open dashboards with near real-time metrics; automate notifications and API access for partners."
    },
    "OPEN USE": {
        "Low":   "Define minimum shared-use licenses for methods and outputs; clarify what can be reused and by whom.",
        "Medium":"Modularize: separate patentable from open components to maximize flexibility and adoption.",
        "High":  "Promote open-source / open hardware where feasible and provide replication kits for other territories."
    }
}
FALLBACK_LOW    = "Strengthen foundational collaboration mechanisms; set minimum open standards."
FALLBACK_MEDIUM = "Standardize what already works (playbooks, templates) and extend it to more partners."
FALLBACK_HIGH   = "Protect and scale what works: codify practices, mentor new teams, and measure outcomes at larger scope."

def get_recommendation(dim: str, cls: str) -> str:
    dim = str(dim).strip()
    cls = str(cls).strip().title()  # "low" -> "Low"
    recs_dim = RECS.get(dim, {})
    if cls in recs_dim:
        return recs_dim[cls]
    if cls == "Low": return FALLBACK_LOW
    if cls == "Medium": return FALLBACK_MEDIUM
    if cls == "High": return FALLBACK_HIGH
    return "No recommendation available."

# =============================================================================
# Utilities (shared)
# =============================================================================
def classify_openness(x: float, low_max: float, med_max: float) -> str:
    if np.isnan(x): return "N/A"
    if x < low_max:       return "Low"
    elif x <= med_max:    return "Medium"
    else:                 return "High"

def preserve_order(seq):
    return list(dict.fromkeys(seq))

# ----- Chart utilities (compact & pretty) -----
GREY_TXT  = "#9CA3AF"
GRID_GREY = "#E5E7EB"

def _multiline(s: str) -> str:
    return "\n".join(s.replace("_", " ").split())

def plot_polar_bar(dimensions_df, size_px=560):
    labels  = dimensions_df["Dimension"].tolist()
    values  = dimensions_df["MeanScore"].tolist()
    classes = dimensions_df["Classification"].tolist()
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    color_map = {"Low": "#DC6B67", "Medium": "#E6B85C", "High": "#6BBF8E", "N/A": "#9CA3AF"}
    colors = [color_map.get(c, "#6B7280") for c in classes]
    labels_ml = [_multiline(s) for s in labels]

    dpi = 120
    size_in = size_px / dpi
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi, subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(np.linspace(0, 2*np.pi, 360), [3]*360, width=np.deg2rad(1), bottom=0,
           color=GREY_TXT, alpha=0.06, edgecolor=None, zorder=0)

    ax.set_ylim(0, 5.0)
    ax.yaxis.grid(color=GRID_GREY, alpha=0.45, linestyle="--")
    ax.xaxis.grid(color=GRID_GREY, alpha=0.35)
    ax.set_yticks([1,2,3,4,5])
    ax.set_yticklabels(["1","2","3","4","5"], color=GREY_TXT, fontsize=9)

    width = 2*np.pi/N * 0.80
    ax.bar(angles, values, width=width, bottom=0,
           color=colors, alpha=0.88, edgecolor="white", linewidth=1.0, zorder=2)

    ax.set_xticks(angles); ax.set_xticklabels([])
    r_label = 4.70
    for ang, lab in zip(angles, labels_ml):
        ax.text(ang, r_label, lab, ha="center", va="center",
                fontsize=9, fontweight="bold", color=PRIMARY)

    ax.set_title("Openness Polar Bar", va='bottom', fontsize=12,
                 fontweight='bold', color=PRIMARY, pad=6)
    fig.tight_layout(pad=0.4)
    return fig

def plot_radar(dimensions_df, size_px=560):
    values = dimensions_df["MeanScore"].to_numpy(dtype=float)
    labels = dimensions_df["Dimension"].tolist()
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    values_closed = np.concatenate([values, values[:1]])
    angles_closed = np.concatenate([angles, angles[:1]])

    dpi = 120
    size_in = size_px / dpi
    fig = plt.figure(figsize=(size_in, size_in), dpi=dpi)
    ax = fig.add_subplot(111, projection='polar')
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(angles_closed, values_closed, linewidth=2, color=PRIMARY)
    ax.fill(angles_closed, values_closed, alpha=0.12, color=PRIMARY)

    ax.set_ylim(1, 5.0)
    ax.set_yticks([1,2,3,4,5])
    ax.set_yticklabels(["1","2","3","4","5"], color=GREY_TXT, fontsize=9)
    ax.yaxis.grid(color=GRID_GREY, alpha=0.45, linestyle="--")
    ax.xaxis.grid(color=GRID_GREY, alpha=0.35)

    labels_ml = [_multiline(s) for s in labels]
    ax.set_xticks(angles); ax.set_xticklabels([])
    r_label = 4.70
    for ang, lab in zip(angles, labels_ml):
        ax.text(ang, r_label, lab, ha="center", va="center",
                fontsize=9, fontweight="bold", color=PRIMARY)

    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_title("Openness Radar", va='bottom', fontsize=12,
                 fontweight='bold', color=PRIMARY, pad=6)
    fig.tight_layout(pad=0.4)
    return fig

# =============================================================================
# MODE B) 2-Axis Matrix (1–3) — with anti-overlap label placement
# =============================================================================
def collect_scores_en(dim_name, questions, key_prefix):
    st.header(dim_name)
    scores = []
    for i, (question, options) in enumerate(questions):
        choice = st.radio(f"{i+1}. {question}", list(options.keys()), key=f"{key_prefix}_{i}")
        scores.append(options[choice])

    st.markdown("**How was this dimension organized structurally?**")
    struct_labels = [
        "Rules and decisions were imposed by external or hierarchical actors",
        "It was a mix between external rules and self-coordination mechanisms",
        "The structure was generated autonomously by the involved actors"
    ]
    struct_values = {struct_labels[0]: 1, struct_labels[1]: 2, struct_labels[2]: 3}
    selected_struct = st.radio("Degree of regulation vs. autonomy", struct_labels, key=f"struct_{key_prefix}")
    struct = struct_values[selected_struct]
    return scores, struct

Q_ENGAGEMENT = [
    ("How many distinct sectors actively participated in the project network?", {
        "5 or more sectors": 3, "4 sectors": 2.5, "3 sectors": 2, "2 sectors": 1.5, "1 or none": 1
    }),
    ("How many actors were actively involved in the project?", {
        "More than 5 actors": 3, "4 actors": 2.5, "3 actors": 2, "2 actors": 1.5, "1 actor or none": 1
    }),
    ("How would you describe the project governance?", {
        "Clear and effective governance": 3, "Mostly effective governance": 2.5,
        "Moderate, with issues": 2, "Unclear or weak": 1.5, "No governance": 1
    }),
    ("How well were ideas listened to among actors?", {
        "Consistent active listening": 3, "Good listening capacity": 2.5,
        "Moderate": 2, "Limited": 1.5, "No listening": 1
    }),
    ("How did the project adapt to changes?", {
        "High adaptability": 3, "Good adaptation": 2.5, "Moderate": 2, "Limited": 1.5, "None": 1
    }),
    ("How did the project respond to unexpected challenges?", {
        "Immediate and effective response": 3, "Generally fast": 2.5,
        "Moderate": 2, "Slow and not effective": 1.5, "No response": 1
    }),
]
Q_APPLICATION = [
    ("How accessible is the generated knowledge?", {
        "Free and unrestricted": 3, "Mostly free": 2.5, "With significant restrictions": 2,
        "Only with specific permissions": 1.5, "Closed": 1
    }),
    ("Is the use, modification or redistribution allowed and facilitated?", {
        "No restrictions": 3, "Minor restrictions": 2.5, "Significant restrictions": 2,
        "Use only, no modification": 1.5, "Not allowed": 1
    }),
    ("How did actors participate in the solution design?", {
        "Participation in all stages": 3, "Frequent, structured participation": 2.5,
        "Moderate with partial structure": 2, "Minimal with weak structure": 1.5, "No participation": 1
    }),
]
Q_INFRA = [
    ("What level of trust and transparency existed among partners?", {
        "High trust and clear transparency": 3, "Generally good": 2.5,
        "Moderate": 2, "Low trust or opaque": 1.5, "No trust/opaque": 1
    }),
    ("How would you describe shared resources, rules, and tools?", {
        "Shared; explicit rules; common toolset": 3, "Mostly shared; some explicit rules": 2.5,
        "Partially shared": 2, "Sparsely shared; weak rules": 1.5, "No shared resources/rules": 1
    }),
    ("How would you describe the coordination mechanisms?", {
        "Formalized and effective": 3, "Mostly formalized": 2.5,
        "Moderate": 2, "Sparsely formalized": 1.5, "No formal coordination": 1
    }),
]
Q_RESEARCH = [
    ("How structured was participation in the research process?", {
        "Broad and structured": 3, "Structured but not constant": 2.5,
        "Moderate": 2, "Limited": 1.5, "None": 1
    }),
    ("How were contributions recognized in the research process?", {
        "Formal and explicit recognition": 3, "Present but not systematic": 2.5,
        "Occasional": 2, "Minimal": 1.5, "No recognition": 1
    }),
    ("Was there a clear strategy for open research?", {
        "Fully defined and implemented": 3, "Mostly defined": 2.5,
        "Partially developed": 2, "Weak strategy": 1.5, "No strategy": 1
    }),
]

def interpret_band(x: float):
    if x >= 2.5: return "High"
    elif x >= 2.0: return "Medium"
    elif x >= 1.5: return "Low"
    else: return "Closed"

def annotate_clustered_points(ax, positions, r_points=18):
    groups = defaultdict(list)
    for name, (x, y) in positions.items():
        groups[(round(x, 3), round(y, 3))].append(name)
    for (x, y), names in groups.items():
        k = len(names)
        if k == 1:
            txt = ax.annotate(
                names[0], xy=(x, y), xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#6B7280"),
                fontsize=9, color=PRIMARY
            )
            txt.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
        else:
            angles = np.linspace(0, 2*np.pi, k, endpoint=False)
            for name, ang in zip(names, angles):
                dx = r_points * np.cos(ang)
                dy = r_points * np.sin(ang)
                txt = ax.annotate(
                    name, xy=(x, y), xytext=(dx, dy), textcoords='offset points',
                    ha='center', va='center', fontsize=9, color=PRIMARY,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#6B7280",
                                    shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.1")
                )
                txt.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])

# =============================================================================
# APP ENTRY — Sidebar mode switch
# =============================================================================
st.title("Openness Assessment")
st.caption("Two analysis modes: (A) Questionnaire 1–5; (B) 2-Axis Matrix 1–3")

mode = st.sidebar.selectbox("Analysis mode", ["A) Questionnaire (1–5)", "B) 2-Axis Matrix (1–3)"])

# =============================================================================
# MODE A (with Polar Bar default)
# =============================================================================
def render_mode_A():
    st.markdown("<div class='section'><div class='block-title'>Questionnaire (1–5)</div>", unsafe_allow_html=True)

    # thresholds + chart options
    low_max = st.sidebar.number_input("Low if mean <", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="low_A")
    med_max = st.sidebar.number_input("Medium if mean ≤", min_value=1.0, max_value=5.0, value=3.5, step=0.1, key="med_A")
    chart_type = st.sidebar.selectbox("Chart type", ["Polar Bar", "Radar"], key="chart_A")  # Polar Bar default
    chart_px   = st.sidebar.slider("Chart width (px)", 420, 900, 560, 10, key="chart_px_A")
    if med_max < low_max:
        st.sidebar.warning("`Medium ≤` should be ≥ `Low <`. Adjust thresholds.")

    data = pd.DataFrame(APERTO_ITEMS, columns=APERTO_COLS)
    with st.expander("Embedded items (preview)"):
        st.dataframe(data.head(10), use_container_width=True)

    dim_order = preserve_order(data["Dimension of openness"].tolist())
    variable_scores = {}
    total_qs = len(data); answered = 0

    for dim in dim_order:
        subset = data[data["Dimension of openness"] == dim]
        if subset.empty: continue
        st.markdown(f"### {dim}")
        cols = st.columns(3)
        for i, row in subset.reset_index(drop=True).iterrows():
            sel = cols[i % 3]
            var_code = str(row["Variable"]).strip()
            q_text   = str(row["Questions"]).strip()
            left_1   = str(row["Scale_1_text"]).strip()
            right_5  = str(row["Scale_5_text"]).strip()

            value = sel.slider(f"{var_code}: {q_text}", 1, 5, 3, 1, key=f"A_{var_code}")
            variable_scores[var_code] = int(value); answered += 1

            lcol, rcol = sel.columns(2)
            lcol.caption(f"1: {left_1}" if left_1 else "1")
            rcol.caption(f"5: {right_5}" if right_5 else "5")

    st.progress(int((answered / total_qs) * 100), text=f"{int((answered / total_qs) * 100)}% completed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    # results
    variables_df = data.assign(
        Score=data["Variable"].map(variable_scores).astype(float),
        Dimension=data["Dimension of openness"],
        Question=data["Questions"]
    )[["Variable","Dimension","Question","Score"]]

    dim_means = variables_df.groupby("Dimension", sort=False)["Score"].mean().reindex(dim_order)
    dimensions_df = pd.DataFrame({
        "Dimension": dim_order,
        "MeanScore": [round(dim_means.get(d, np.nan), 3) for d in dim_order],
        "n_Variables": [int((variables_df["Dimension"] == d).sum()) for d in dim_order]
    })
    dimensions_df["Classification"] = dimensions_df["MeanScore"].apply(lambda x: classify_openness(x, low_max, med_max))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Variable scores")
        st.dataframe(variables_df, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Dimension scores")
        st.dataframe(dimensions_df, use_container_width=True, hide_index=True)

    # visualization (always fully visible)
    st.markdown("<h3 class='center-title'>Visualization</h3>", unsafe_allow_html=True)
    if chart_type == "Polar Bar":
        fig = plot_polar_bar(dimensions_df, size_px=chart_px)
    else:
        fig = plot_radar(dimensions_df, size_px=chart_px)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    st.image(png_buf.getvalue(), width=chart_px)
    st.download_button(
        label=f"Download {chart_type} (PNG)",
        data=png_buf.getvalue(),
        file_name=f"openness_{chart_type.lower().replace(' ','_')}_{date.today()}.png",
        mime="image/png",
        type="secondary"
    )

    # -----------------------------
    # Recommendations (NEW SYSTEM)
    # -----------------------------
    st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
    rec_rows = []
    for _, row in dimensions_df.iterrows():
        dim = row["Dimension"]; cls = row["Classification"]
        rec = get_recommendation(dim, cls)
        rec_rows.append({"Dimension": dim, "Classification": cls, "Recommendation": rec})
    recommendations_df = pd.DataFrame(rec_rows, columns=["Dimension","Classification","Recommendation"])

    order = {"Low":0, "Medium":1, "High":2, "N/A":3}
    recommendations_df = recommendations_df.sort_values(["Classification","Dimension"], key=lambda s: s.map(order))

    for _, r in recommendations_df.iterrows():
        cls = str(r["Classification"]).lower()
        card_class = "card-low" if cls=="low" else "card-medium" if cls=="medium" else "card-high"
        st.markdown(
            f"""
            <div class="card {card_class}">
              <div class="block-title">{r['Dimension']} — {r['Classification']}</div>
              <div>{r['Recommendation']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # CSV downloads
    st.markdown("#### Downloads")
    vbuf = io.StringIO(); variables_df.to_csv(vbuf, index=False)
    dbuf = io.StringIO(); dimensions_df.to_csv(dbuf, index=False)
    rbuf = io.StringIO(); recommendations_df.to_csv(rbuf, index=False)
    d1, d2, d3 = st.columns(3)
    with d1: st.download_button("Variable scores (CSV)", vbuf.getvalue(), f"openness_variable_scores_{date.today()}.csv", "text/csv", key="vA")
    with d2: st.download_button("Dimension scores (CSV)", dbuf.getvalue(), f"openness_dimension_scores_{date.today()}.csv", "text/csv", key="dA")
    with d3: st.download_button("Recommendations (CSV)", rbuf.getvalue(), f"openness_recommendations_{date.today()}.csv", "text/csv", key="rA")

# =============================================================================
# MODE B
# =============================================================================
def render_mode_B():
    st.markdown("### 2-Axis Matrix: Level of Openness vs. Regulation/Autonomy (1–3 scale)")
    st.markdown("For each dimension, rate items on a 1–3 scale (0.5 steps in options). Then place each dimension on the matrix: **X = Level of Openness (1=Low, 2=Medium, 3=High)**; **Y = Regulation vs. Autonomy (1=More Regulated/Exogenous, 2=Mixed, 3=More Autonomous/Endogenous)**.")

    st.divider()
    s1, y1 = collect_scores_en("1. Open Engagement", Q_ENGAGEMENT, "B_collab")
    st.divider()
    s2, y2 = collect_scores_en("2. Open Application & Adaptation", Q_APPLICATION, "B_app")
    st.divider()
    s3, y3 = collect_scores_en("3. Open Interaction Infrastructure", Q_INFRA, "B_infra")
    st.divider()
    s4, y4 = collect_scores_en("4. Open Research", Q_RESEARCH, "B_research")

    if st.button("Compute scores (Matrix)"):
        def avg(lst): return round(float(np.mean(lst)), 2) if len(lst) else np.nan

        x_scores = {"Engagement": avg(s1), "Application": avg(s2), "Interaction": avg(s3), "Research": avg(s4)}
        y_scores = {"Engagement": y1,     "Application": y2,     "Interaction": y3,     "Research": y4}

        dfB = pd.DataFrame({
            "Dimension": list(x_scores.keys()),
            "X_LevelOpenness(1-3)": [x_scores[k] for k in x_scores],
            "Y_Regulation→Autonomy(1-3)": [y_scores[k] for k in x_scores],
        })
        dfB["Band"] = dfB["X_LevelOpenness(1-3)"].apply(interpret_band)

        st.subheader("Matrix scores")
        st.dataframe(dfB, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_facecolor("white")
        for v in [1, 2, 3]:
            ax.axvline(v, color="#E5E7EB", lw=1, zorder=0)
            ax.axhline(v, color="#E5E7EB", lw=1, zorder=0)

        dim_positions = {
            "Engagement":  (x_scores["Engagement"],  y_scores["Engagement"]),
            "Application": (x_scores["Application"], y_scores["Application"]),
            "Interaction": (x_scores["Interaction"], y_scores["Interaction"]),
            "Research":    (x_scores["Research"],    y_scores["Research"]),
        }
        for (x, y) in dim_positions.values():
            ax.scatter(x, y, color="steelblue", s=180, edgecolors="black", zorder=3)

        annotate_clustered_points(ax, dim_positions, r_points=18)

        ax.set_xlim(0.8, 3.2); ax.set_ylim(0.8, 3.2)
        ax.set_xticks([1, 2, 3]); ax.set_yticks([1, 2, 3])
        ax.set_xticklabels(["Low (1)", "Medium (2)", "High (3)"])
        ax.set_yticklabels(["More Regulated (1)", "Mixed (2)", "More Autonomous (3)"])
        ax.set_xlabel("Level of Openness")
        ax.set_ylabel("Regulation vs. Autonomy")
        ax.set_title("2-Axis Matrix: Openness vs. Regulation", color=PRIMARY)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        st.pyplot(fig)

        st.markdown("#### Interpretation")
        for _, r in dfB.iterrows():
            st.markdown(f"- **{r['Dimension']}**: Openness={r['X_LevelOpenness(1-3)']} → **{r['Band']}**, Autonomy={r['Y_Regulation→Autonomy(1-3)']}.")

# --------- Route switch ----------
if mode.startswith("A"):
    render_mode_A()
else:
    render_mode_B()

st.divider()
st.markdown('*Copyright (C) 2025. CIRAD*')
st.caption('**Authors: Alejandro Taborda A., (latabordaa@unal.edu.co), Chloé Lecombe (chloe.lecomte@cirad.fr)**')
st.image('aperto.png', width=250)
