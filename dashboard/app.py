from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="CareerPath AI Dashboard",
    page_icon="📊",
    layout="wide",
)


DATA_PATH = Path("job_featured.csv")


@st.cache_data(show_spinner="Memuat dataset CareerPath AI...")
def load_data(path: Path) -> pd.DataFrame:
    """Memuat dataset utama dan melakukan penyesuaian tipe data ringan."""
    if not path.exists():
        raise FileNotFoundError(f"File {path.name} tidak ditemukan.")

    data = pd.read_csv(path)

    for column in ["listed_time", "original_listed_time"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    if "posted_month" not in data.columns:
        date_column = None
        if "listed_time" in data.columns:
            date_column = "listed_time"
        elif "original_listed_time" in data.columns:
            date_column = "original_listed_time"

        if date_column:
            data["posted_month"] = data[date_column].dt.month
            data["posted_year"] = data[date_column].dt.year

    text_defaults = {
        "title": "not_specified",
        "location": "unknown",
        "experience_level_clean": "unknown",
        "formatted_work_type": "unknown",
        "dominant_skill": "not_specified",
        "skill_text": "",
        "city_clean": "unknown",
    }

    for column, default_value in text_defaults.items():
        if column not in data.columns:
            data[column] = default_value
        data[column] = data[column].fillna(default_value).astype(str)

    if "is_remote" not in data.columns:
        if "remote_allowed" in data.columns:
            data["is_remote"] = pd.to_numeric(data["remote_allowed"], errors="coerce").fillna(0).astype(int)
        else:
            data["is_remote"] = 0
    data["is_remote"] = np.where(pd.to_numeric(data["is_remote"], errors="coerce").fillna(0) == 1, 1, 0)

    if "desc_length" not in data.columns:
        description_column = "description_clean" if "description_clean" in data.columns else "description"
        if description_column in data.columns:
            data["desc_length"] = data[description_column].fillna("").astype(str).str.split().str.len()
        else:
            data["desc_length"] = 0

    if "total_skills" not in data.columns:
        data["total_skills"] = np.where(data["dominant_skill"].str.lower().ne("not_specified"), 1, 0)

    return data


def normalize_label(value: str) -> str:
    """Merapikan label untuk tampilan dashboard."""
    value = str(value).replace("_", " ").strip()
    return value.title() if value else "Unknown"


def value_counts_frame(data: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """Membuat dataframe frekuensi untuk visualisasi."""
    if column not in data.columns or data.empty:
        return pd.DataFrame(columns=[column, "jumlah"])

    counts = (
        data[column]
        .fillna("unknown")
        .astype(str)
        .replace("", "unknown")
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = [column, "jumlah"]
    return counts


def top_skill_frame(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Menghitung top skill dari skill_text jika tersedia, fallback ke dominant_skill."""
    if data.empty:
        return pd.DataFrame(columns=["skill", "jumlah"])

    if "dominant_skill" in data.columns:
        skill_series = data["dominant_skill"].fillna("not_specified").astype(str)
        skill_series = skill_series[skill_series.ne("not_specified")]
    elif "skill_text" in data.columns and data["skill_text"].fillna("").str.strip().ne("").any():
        skill_series = (
            data["skill_text"]
            .fillna("")
            .astype(str)
            .str.split()
            .explode()
            .str.strip()
        )
        skill_series = skill_series[skill_series.ne("") & skill_series.ne("not_specified")]
    else:
        return pd.DataFrame(columns=["skill", "jumlah"])

    skill_series.name = "skill"
    return skill_series.value_counts().head(top_n).reset_index(name="jumlah").rename(columns={"index": "skill"})


def render_empty_state(message: str) -> None:
    """Menampilkan pesan saat data tidak tersedia untuk grafik."""
    st.warning(message)


try:
    df = load_data(DATA_PATH)
except FileNotFoundError as error:
    st.error(
        "Dataset utama tidak ditemukan. Pastikan `job_featured.csv` berada di folder yang sama dengan `app.py`."
    )
    st.caption(str(error))
    st.stop()
except Exception as error:
    st.error("Terjadi error saat memuat dataset.")
    st.exception(error)
    st.stop()


st.title("CareerPath AI Dashboard")
st.subheader("Analisis Pasar Kerja & Tren Skill")
st.caption(
    "Dashboard interaktif untuk membaca peluang kerja, distribusi skill, level pengalaman, dan tren posting lowongan."
)

st.divider()


with st.sidebar:
    st.header("Filter Dashboard")
    st.caption("Gunakan filter berikut untuk menyesuaikan seluruh grafik, KPI, insight, dan tabel.")

    location_options = sorted(df["location"].dropna().astype(str).unique().tolist())
    selected_locations = st.multiselect("Lokasi", location_options)

    experience_options = sorted(df["experience_level_clean"].dropna().astype(str).unique().tolist())
    selected_experiences = st.multiselect("Experience Level", experience_options)

    work_type_options = sorted(df["formatted_work_type"].dropna().astype(str).unique().tolist())
    selected_work_types = st.multiselect("Work Type", work_type_options)

    remote_option = st.selectbox(
        "Remote / Non Remote",
        ["Semua", "Remote", "Non Remote / Unknown"],
    )

    search_title = st.text_input("Search Job Title", placeholder="Contoh: data analyst, sales, engineer")

    st.divider()
    st.caption(f"Total data awal: {len(df):,} lowongan")


filtered_df = df.copy()

if selected_locations:
    filtered_df = filtered_df[filtered_df["location"].isin(selected_locations)]

if selected_experiences:
    filtered_df = filtered_df[filtered_df["experience_level_clean"].isin(selected_experiences)]

if selected_work_types:
    filtered_df = filtered_df[filtered_df["formatted_work_type"].isin(selected_work_types)]

if remote_option == "Remote":
    filtered_df = filtered_df[filtered_df["is_remote"] == 1]
elif remote_option == "Non Remote / Unknown":
    filtered_df = filtered_df[filtered_df["is_remote"] == 0]

if search_title.strip():
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(search_title.strip(), case=False, na=False)
    ]


if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter saat ini. Silakan ubah pilihan filter.")
    st.stop()


with st.container():
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

    total_jobs = len(filtered_df)
    total_locations = filtered_df["location"].nunique()
    total_unique_skills = (
        top_skill_frame(filtered_df, top_n=10_000)["skill"].nunique()
        if {"dominant_skill", "skill_text"}.intersection(filtered_df.columns)
        else 0
    )
    avg_desc_length = filtered_df["desc_length"].mean()

    kpi_1.metric("Total Lowongan", f"{total_jobs:,}")
    kpi_2.metric("Total Lokasi", f"{total_locations:,}")
    kpi_3.metric("Total Skill Unik", f"{total_unique_skills:,}")
    kpi_4.metric("Rata-rata Panjang Deskripsi", f"{avg_desc_length:,.0f} kata")


st.divider()


st.header("Visualisasi Utama")

chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
    st.subheader("Top 10 Job Title")
    top_titles = value_counts_frame(filtered_df, "title", 10)
    if top_titles.empty:
        render_empty_state("Kolom title tidak tersedia untuk visualisasi.")
    else:
        fig = px.bar(
            top_titles.sort_values("jumlah"),
            x="jumlah",
            y="title",
            orientation="h",
            text="jumlah",
            color_discrete_sequence=["#2F6B9A"],
            labels={"jumlah": "Jumlah Lowongan", "title": "Job Title"},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

with chart_col_2:
    st.subheader("Top 10 Location")
    top_locations = value_counts_frame(filtered_df, "location", 10)
    if top_locations.empty:
        render_empty_state("Kolom location tidak tersedia untuk visualisasi.")
    else:
        fig = px.bar(
            top_locations,
            x="location",
            y="jumlah",
            text="jumlah",
            color_discrete_sequence=["#3B8C66"],
            labels={"jumlah": "Jumlah Lowongan", "location": "Lokasi"},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


chart_col_3, chart_col_4 = st.columns(2)

with chart_col_3:
    st.subheader("Distribusi Experience Level")
    exp_counts = value_counts_frame(filtered_df, "experience_level_clean", 20)
    if exp_counts.empty:
        render_empty_state("Kolom experience_level_clean tidak tersedia untuk visualisasi.")
    else:
        exp_counts["experience_level_clean"] = exp_counts["experience_level_clean"].apply(normalize_label)
        fig = px.pie(
            exp_counts,
            names="experience_level_clean",
            values="jumlah",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

with chart_col_4:
    st.subheader("Remote vs Non Remote")
    remote_counts = filtered_df["is_remote"].map({1: "Remote", 0: "Non Remote / Unknown"}).value_counts().reset_index()
    remote_counts.columns = ["status_remote", "jumlah"]
    fig = px.pie(
        remote_counts,
        names="status_remote",
        values="jumlah",
        hole=0.45,
        color_discrete_sequence=["#4C78A8", "#F58518"],
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


chart_col_5, chart_col_6 = st.columns(2)

with chart_col_5:
    st.subheader("Top Skill")
    top_skills = top_skill_frame(filtered_df, 10)
    if top_skills.empty:
        render_empty_state("Kolom dominant_skill atau skill_text tidak tersedia untuk visualisasi skill.")
    else:
        fig = px.bar(
            top_skills.sort_values("jumlah"),
            x="jumlah",
            y="skill",
            orientation="h",
            text="jumlah",
            color_discrete_sequence=["#7A5C99"],
            labels={"jumlah": "Jumlah Kemunculan", "skill": "Skill"},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

with chart_col_6:
    st.subheader("Tren Posting Lowongan")
    trend_df = filtered_df.copy()

    if "listed_time" in trend_df.columns and trend_df["listed_time"].notna().any():
        trend_df["posting_period"] = trend_df["listed_time"].dt.to_period("M").dt.to_timestamp()
    elif {"posted_year", "posted_month"}.issubset(trend_df.columns):
        trend_df["posting_period"] = pd.to_datetime(
            trend_df["posted_year"].astype("Int64").astype(str)
            + "-"
            + trend_df["posted_month"].astype("Int64").astype(str)
            + "-01",
            errors="coerce",
        )
    else:
        trend_df["posting_period"] = pd.NaT

    trend_counts = (
        trend_df.dropna(subset=["posting_period"])
        .groupby("posting_period")
        .size()
        .reset_index(name="jumlah")
        .sort_values("posting_period")
    )

    if trend_counts.empty:
        render_empty_state("Kolom waktu posting tidak tersedia untuk visualisasi tren.")
    else:
        fig = px.line(
            trend_counts,
            x="posting_period",
            y="jumlah",
            markers=True,
            labels={"posting_period": "Periode Posting", "jumlah": "Jumlah Lowongan"},
        )
        fig.update_traces(line_color="#2F6B9A")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


st.divider()


st.header("Insight Otomatis")

insight_col_1, insight_col_2 = st.columns(2)

top_location_text = (
    filtered_df["location"].value_counts().idxmax()
    if "location" in filtered_df.columns and not filtered_df.empty
    else "tidak tersedia"
)
top_title_text = (
    filtered_df["title"].value_counts().idxmax()
    if "title" in filtered_df.columns and not filtered_df.empty
    else "tidak tersedia"
)
top_experience_text = (
    normalize_label(filtered_df["experience_level_clean"].value_counts().idxmax())
    if "experience_level_clean" in filtered_df.columns and not filtered_df.empty
    else "tidak tersedia"
)
top_skill_text = top_skills.iloc[0]["skill"] if not top_skills.empty else "tidak tersedia"

with insight_col_1:
    st.info(f"Lokasi paling aktif berdasarkan filter saat ini adalah **{top_location_text}**.")
    st.info(f"Job title paling banyak muncul adalah **{top_title_text}**.")

with insight_col_2:
    st.success(f"Mayoritas lowongan berada pada level **{top_experience_text}**.")
    st.success(f"Skill paling sering muncul adalah **{top_skill_text}**.")

remote_share = filtered_df["is_remote"].mean() * 100
if remote_share >= 20:
    st.info(f"Porsi lowongan remote pada filter ini mencapai **{remote_share:.1f}%**, cukup menarik untuk kandidat yang mencari fleksibilitas kerja.")
else:
    st.info(f"Porsi lowongan remote pada filter ini sebesar **{remote_share:.1f}%**, sehingga mayoritas peluang masih non-remote atau tidak mencantumkan opsi remote.")


st.divider()


st.header("Tabel Data Interaktif")

table_columns = [
    "title",
    "location",
    "experience_level_clean",
    "formatted_work_type",
    "dominant_skill",
]
available_table_columns = [column for column in table_columns if column in filtered_df.columns]

if len(filtered_df) < 10:
    row_count = len(filtered_df)
    st.caption(f"Menampilkan seluruh {row_count} baris hasil filter.")
else:
    row_count = st.slider(
        "Jumlah baris yang ditampilkan",
        min_value=10,
        max_value=min(500, len(filtered_df)),
        value=min(50, len(filtered_df)),
        step=10,
    )

st.dataframe(
    filtered_df[available_table_columns].head(row_count),
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Unduh Data Hasil Filter",
    data=csv_data,
    file_name="careerpath_ai_filtered_jobs.csv",
    mime="text/csv",
)


st.divider()
st.caption(
    "CareerPath AI Dashboard | Dataset: job_featured.csv | Siap untuk analisis pasar kerja, dashboard Streamlit, dan pengembangan recommendation system."
)
