import streamlit as st
import requests
from streamlit_lottie import st_lottie

# --------------- Helper Functions -------------------

def calculate_sgpa(marks, credits):
    total_credits = sum(credits)
    total_weighted_points = 0

    for i, mark in enumerate(marks):
        if mark >= 90:
            grade = 10
        elif mark >= 80:
            grade = 9
        elif mark >= 70:
            grade = 8
        elif mark >= 60:
            grade = 7
        elif mark >= 55:
            grade = 6
        elif mark >= 50:
            grade = 5
        elif mark >= 40:
            grade = 4
        else:
            grade = 0

        total_weighted_points += grade * credits[i]

    sgpa = total_weighted_points / total_credits
    return sgpa

def cgpa_to_percentage(cgpa, conversion_formula):
    if conversion_formula == "VTU (CGPA * 10)":
        return cgpa * 10
    elif conversion_formula == "CGPA * 9.5":
        return cgpa * 9.5
    elif conversion_formula == "CGPA * 9.0":
        return cgpa * 9.0
    else:
        return cgpa * conversion_formula

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --------------- Streamlit App -------------------

# Set page config
st.set_page_config(
    page_title="SGPA Calculator",
    page_icon="📊",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
<style>
body, .main {
    background-color: #0e1117;
    color: #e1e1e1;
}
[data-testid="stTabs"] button {
    background-color: #1f2937;
    color: #ffffff;
    font-weight: bold;
    border-radius: 6px;
    margin-right: 6px;
    transition: background-color 0.3s ease;
}
[data-testid="stTabs"] button:hover {
    background-color: #3b82f6;
}
.stButton>button {
    background-color: #3b82f6;
    color: white;
    font-weight: 600;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #2563eb;
}
input, textarea {
    background-color: #1e1e1e !important;
    color: #f1f1f1 !important;
}
h1, h2, h3 {
    color: #f8fafc;
}
.footer {
    color: #9ca3af;
    font-size: 0.9rem;
    padding-top: 2rem;
    border-top: 1px solid #374151;
}

/* Remove white background container */
main > div.block-container {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# --------------- Sidebar -------------------

with st.sidebar:
    st.title("📘 SGPA Toolkit")
    st.markdown("Use this tool to calculate SGPA, CGPA, and convert to percentage.")
    st.markdown("Powered by **Shashank Industries**")
    st.markdown("Created by Shashank V")
    lottie_anim = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_pwohahvd.json")
    if lottie_anim:
        st_lottie(lottie_anim, height=150)

# --------------- Main Title -------------------

st.title("📊 SGPA & CGPA Calculator")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧮 SGPA Calculator", "📚 SGPA ➡️ CGPA", "🎯 CGPA ➡️ Percentage"])

# ---------------- SGPA Tab ------------------

with tab1:
    st.header("🧮 SGPA Calculator")

    with st.form("sgpa_form"):
        num_subjects = st.number_input("📘 Number of Subjects", min_value=1, max_value=15, value=8)
        marks = []
        credits = []

        for i in range(num_subjects):
            col1, col2 = st.columns([3, 2])
            with col1:
                mark = st.number_input(f"Marks for Subject {i+1}", min_value=0.0, max_value=100.0, key=f"mark_{i}")
                marks.append(mark)
            with col2:
                credit = st.number_input(f"Credits for Subject {i+1}", min_value=0.0, max_value=10.0, key=f"credit_{i}")
                credits.append(credit)

        submitted = st.form_submit_button("🚀 Calculate SGPA")

        if submitted:
            if any(m > 100 or m < 0 for m in marks):
                st.error("❌ Marks must be between 0 and 100!")
            else:
                sgpa = calculate_sgpa(marks, credits)
                st.success(f"✅ Your SGPA is: **{sgpa:.2f}**")

                if sgpa >= 9.0:
                    st.info("Excellent performance! 🎉")
                elif sgpa >= 8.0:
                    st.info("Very good performance! 👍")
                elif sgpa >= 7.0:
                    st.info("Good performance! 👏")
                elif sgpa >= 6.0:
                    st.info("Satisfactory performance! ✅")
                elif sgpa >= 5.0:
                    st.info("Passed! 😊")
                else:
                    st.warning("Needs improvement! 📚")

# ---------------- CGPA Tab ------------------

with tab2:
    st.header("📚 Convert SGPA to CGPA")

    with st.form("cgpa_form"):
        num_semesters = st.number_input("🎓 Number of Semesters", min_value=1, max_value=10, value=4)
        sgpa_values = []
        semester_credits = []

        for i in range(num_semesters):
            col1, col2 = st.columns([3, 2])
            with col1:
                sgpa = st.number_input(f"SGPA for Semester {i+1}", min_value=0.0, max_value=10.0, key=f"sgpa_{i}")
                sgpa_values.append(sgpa)
            with col2:
                credit = st.number_input(f"Credits for Semester {i+1}", min_value=0.0, max_value=50.0, key=f"sem_credit_{i}")
                semester_credits.append(credit)

        submitted = st.form_submit_button("📘 Calculate CGPA")

        if submitted:
            if any(s < 0 or s > 10 for s in sgpa_values):
                st.error("❌ SGPA must be between 0 and 10!")
            else:
                total_credits = sum(semester_credits)
                weighted_sum = sum(sgpa_values[i] * semester_credits[i] for i in range(num_semesters))
                cgpa = weighted_sum / total_credits
                st.success(f"📌 Your CGPA is: **{cgpa:.2f}**")

                if cgpa >= 8.5:
                    st.info("First Class with Distinction! 🎉")
                elif cgpa >= 7.0:
                    st.info("First Class! 👍")
                elif cgpa >= 6.0:
                    st.info("Second Class! 👏")
                elif cgpa >= 5.0:
                    st.info("Pass Class! ✅")
                else:
                    st.warning("Needs improvement! 📚")

# ---------------- Percentage Tab ------------------

with tab3:
    st.header("🎯 Convert CGPA to Percentage")

    with st.form("percentage_form"):
        cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, value=8.0)
        formula_option = st.selectbox(
            "Choose Conversion Formula",
            ["VTU (CGPA * 10)", "CGPA * 9.5", "CGPA * 9.0", "Custom multiplier"]
        )

        multiplier = 10.0
        if formula_option == "Custom multiplier":
            multiplier = st.number_input("Enter custom multiplier", min_value=0.0, max_value=15.0, value=10.0)

        submitted = st.form_submit_button("📈 Convert to Percentage")

        if submitted:
            if cgpa < 0 or cgpa > 10:
                st.error("❌ CGPA must be between 0 and 10!")
            else:
                if formula_option == "Custom multiplier":
                    percentage = cgpa * multiplier
                else:
                    percentage = cgpa_to_percentage(cgpa, formula_option)

                st.success(f"🎓 Your percentage is: **{percentage:.2f}%**")

                if percentage >= 85.0:
                    st.info("Outstanding performance! 🎉")
                elif percentage >= 75.0:
                    st.info("Excellent performance! 👍")
                elif percentage >= 65.0:
                    st.info("Very good performance! 👏")
                elif percentage >= 55.0:
                    st.info("Good performance! ✅")
                elif percentage >= 45.0:
                    st.info("Passed! 😊")
                else:
                    st.warning("Needs improvement! 📚")
