import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hospital Command Center", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

h1 {
    color: #00E5FF;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
}

h2 {
    color: #FFD700;
}

h3 {
    color: #00FFB3;
}

p {
    color: white;
    font-size: 18px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #232526, #414345);
}

[data-testid="stMetricValue"] {
    color: #00FFB3;
    font-size: 35px;
}

[data-testid="stMetricLabel"] {
    color: white;
    font-size: 18px;
}

.stDataFrame {
    border-radius: 15px;
    overflow: hidden;
}

hr {
    border: 1px solid #00E5FF;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("🏥 Hospital Command Center Dashboard")

st.markdown("""
This dashboard monitors hospital operations including patient journey,
ICU occupancy, billing analytics, emergency room management,
staff allocation, and operation theatre activities.
""")

# ---------- LOAD DATASETS ----------
patient_df = pd.read_csv("data/patient_journey_dataset.csv")
bed_df = pd.read_csv("data/bed_occupancy_dataset.csv")
billing_df = pd.read_csv("data/medical_billing_dataset.csv")
er_df = pd.read_csv("data/emergency_room_dataset.csv")
staff_df = pd.read_csv("data/staff_allocation_dataset.csv")
ot_df = pd.read_csv("data/operation_theatre_dataset.csv")

# ---------- SIDEBAR ----------
menu = st.sidebar.selectbox(
    "Select Dataset",
    [
        "Patient Journey",
        "Bed Occupancy / ICU",
        "Medical Billing",
        "Emergency Room",
        "Staff Allocation",
        "Operation Theatre"
    ]
)

# ---------- PATIENT JOURNEY ----------
if menu == "Patient Journey":

    st.header("🧑‍⚕️ Patient Journey Dataset")

    st.dataframe(patient_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", len(patient_df))
    col2.metric("Emergency Cases", patient_df["Emergency_Flag"].astype(str).value_counts().get("True", 0))
    col3.metric("ICU Admissions", patient_df["ICU_Admission"].astype(str).value_counts().get("True", 0))

    st.divider()

    st.subheader("🏥 Department Wise Patient Count")

    dept_chart = patient_df["Department"].value_counts()
    st.bar_chart(dept_chart, color="#00E5FF")

    st.subheader("🦠 Disease Type Count")

    disease_chart = patient_df["Disease_Type"].value_counts()
    st.bar_chart(disease_chart, color="#FFD700")

    st.subheader("🚨 Emergency Flag Count")

    emergency_chart = patient_df["Emergency_Flag"].value_counts()
    st.bar_chart(emergency_chart, color="#FF4B4B")

# ---------- BED OCCUPANCY ----------
elif menu == "Bed Occupancy / ICU":

    st.header("🛏️ Bed Occupancy / ICU Dataset")

    st.dataframe(bed_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Beds", int(bed_df["Total_Beds"].sum()))
    col2.metric("Occupied Beds", int(bed_df["Occupied_Beds"].sum()))
    col3.metric("Available Beds", int(bed_df["Available_Beds"].sum()))

    st.divider()

    st.subheader("🏥 Ward Wise Count")

    st.bar_chart(bed_df["Ward"].value_counts(), color="#00FFB3")

    st.subheader("🏢 Department Wise Bed Records")

    st.bar_chart(bed_df["Department"].value_counts(), color="#FFB703")

    st.subheader("🚑 Emergency Load Count")

    st.bar_chart(bed_df["Emergency_Load"].value_counts(), color="#FB8500")

# ---------- MEDICAL BILLING ----------
elif menu == "Medical Billing":

    st.header("💰 Medical Billing / Revenue Dataset")

    st.dataframe(billing_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue", int(billing_df["Total_Bill"].sum()))
    col2.metric("Claim Amount", int(billing_df["Claim_Amount"].sum()))
    col3.metric("Outstanding Amount", int(billing_df["Outstanding_Amount"].sum()))

    st.divider()

    st.subheader("💳 Payment Status Count")

    st.bar_chart(billing_df["Payment_Status"].value_counts(), color="#00E5FF")

    st.subheader("📄 Claim Status Count")

    st.bar_chart(billing_df["Claim_Status"].value_counts(), color="#FFD166")

    st.subheader("🏦 Insurance Provider Count")

    st.bar_chart(billing_df["Insurance_Provider"].value_counts(), color="#EF476F")

# ---------- EMERGENCY ROOM ----------
elif menu == "Emergency Room":

    st.header("🚑 Emergency Room Dataset")

    st.dataframe(er_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total ER Cases", len(er_df))
    col2.metric("Average Wait Time", round(er_df["Wait_Time"].mean(), 2))
    col3.metric("Average Treatment Time", round(er_df["Treatment_Time"].mean(), 2))

    st.divider()

    st.subheader("🚨 Severity Level Count")

    st.bar_chart(er_df["Severity_Level"].value_counts(), color="#FF006E")

    st.subheader("🏥 Triage Level Count")

    st.bar_chart(er_df["Triage_Level"].value_counts(), color="#8338EC")

    st.subheader("📋 Admission or Discharge")

    st.bar_chart(er_df["Admission_or_Discharge"].value_counts(), color="#3A86FF")

# ---------- STAFF ALLOCATION ----------
elif menu == "Staff Allocation":

    st.header("👨‍⚕️ Staff Allocation Dataset")

    st.dataframe(staff_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Staff", len(staff_df))
    col2.metric("Patients Assigned", int(staff_df["Patients_Assigned"].sum()))
    col3.metric("Overtime Hours", int(staff_df["Overtime_Hours"].sum()))

    st.divider()

    st.subheader("👨‍⚕️ Staff Role Count")

    st.bar_chart(staff_df["Staff_Role"].value_counts(), color="#06D6A0")

    st.subheader("🏢 Department Wise Staff")

    st.bar_chart(staff_df["Department"].value_counts(), color="#118AB2")

    st.subheader("✅ Availability Count")

    st.bar_chart(staff_df["Availability"].value_counts(), color="#FFD166")

# ---------- OPERATION THEATRE ----------
elif menu == "Operation Theatre":

    st.header("🏥 Operation Theatre Dataset")

    st.dataframe(ot_df)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Surgeries", len(ot_df))
    col2.metric("Average Delay", round(ot_df["Delay_Time"].mean(), 2))
    col3.metric("Maximum Delay", int(ot_df["Delay_Time"].max()))

    st.divider()

    st.subheader("🔪 Surgery Type Count")

    st.bar_chart(ot_df["Surgery_Type"].value_counts(), color="#EF476F")

    st.subheader("🏢 OT Room Usage")

    st.bar_chart(ot_df["OT_Room"].value_counts(), color="#06D6A0")

    st.subheader("👨‍⚕️ Surgeon Wise Surgery Count")

    st.bar_chart(ot_df["Surgeon"].value_counts(), color="#118AB2")

    st.subheader("🛠️ Equipment Usage Count")

    st.bar_chart(ot_df["Equipment_Used"].value_counts(), color="#FFD166")