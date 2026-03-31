import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hektors epiteter", layout="wide")
st.title("Hektors epiteter i Iliaden")

df = pd.read_csv("hector_morphosyntax_v2.csv")
df["Book"] = df["Book"].astype(int)

# ── Sidebar filters ──
st.sidebar.header("Filtre")

all_lemmata = sorted(df["Lemma"].unique())
sel_lemmata = st.sidebar.multiselect("Lemmata", all_lemmata, default=all_lemmata)

book_range = st.sidebar.slider("Sange", 1, 24, (1, 24))

all_pos = sorted(df["POS"].unique())
sel_pos = st.sidebar.multiselect("POS", all_pos, default=all_pos)

all_cases = sorted(df["Case"].dropna().unique())
sel_case = st.sidebar.multiselect("Kasus", all_cases, default=all_cases)

all_conf = sorted(df["Confidence"].unique())
sel_conf = st.sidebar.multiselect("Confidence", all_conf, default=all_conf)

speakers = sorted(df["Speaker"].dropna().unique())
sel_speaker = st.sidebar.multiselect("Speaker", speakers, default=speakers)

# ── Apply filters ──
filt = df[
    (df["Lemma"].isin(sel_lemmata))
    & (df["Book"].between(*book_range))
    & (df["POS"].isin(sel_pos))
    & (df["Case"].isin(sel_case) | df["Case"].isna())
    & (df["Confidence"].isin(sel_conf))
    & (df["Speaker"].isin(sel_speaker) | df["Speaker"].isna())
]

st.metric("Forekomster", len(filt))

# ── Tab layout ──
tab1, tab2, tab3, tab4 = st.tabs([
    "Epiteter pr. sang",
    "Lemma-frekvens",
    "Kasus & POS",
    "Tabel",
])

with tab1:
    counts = filt.groupby(["Book", "Lemma"]).size().reset_index(name="Antal")
    fig = px.bar(
        counts, x="Book", y="Antal", color="Lemma", barmode="group",
        labels={"Book": "Sang", "Antal": "Antal forekomster"},
    )
    fig.update_xaxes(dtick=1, tick0=1)
    fig.update_yaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    freq = filt["Lemma"].value_counts().reset_index()
    freq.columns = ["Lemma", "Antal"]
    fig2 = px.bar(
        freq, x="Antal", y="Lemma", orientation="h",
        labels={"Antal": "Antal forekomster"},
        text="Antal",
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        case_counts = filt["Case"].value_counts().reset_index()
        case_counts.columns = ["Kasus", "Antal"]
        fig3 = px.bar(case_counts, x="Kasus", y="Antal")
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        pos_counts = filt["POS"].value_counts().reset_index()
        pos_counts.columns = ["POS", "Antal"]
        fig4 = px.bar(pos_counts, x="POS", y="Antal")
        st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.dataframe(
        filt[["Book", "Line", "Epithet", "Lemma", "Speaker", "Addressee",
              "Case", "Number", "POS", "Confidence", "Greek"]],
        use_container_width=True,
        height=600,
    )