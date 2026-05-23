import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Hospital Command Center", layout="wide")

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
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #232526, #414345);
}
[data-testid="stMetricValue"] {
    color: #00FFB3;
    font-size: 35px;
}
[data-testid="stMetricLabel"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Command Center Dashboard")

st.markdown("""
This dashboard monitors hospital operations including patient journey, ICU occupancy,
billing analytics, emergency room management, staff allocation, and operation theatre activities.
""")

patient_df = pd.read_csv("data/patient_journey_dataset.csv")
bed_df = pd.read_csv("data/bed_occupancy_dataset.csv")
billing_df = pd.read_csv("data/medical_billing_dataset.csv")
er_df = pd.read_csv("data/emergency_room_dataset.csv")
staff_df = pd.read_csv("data/staff_allocation_dataset.csv")
ot_df = pd.read_csv("data/operation_theatre_dataset.csv")

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

def ml_accuracy(df, feature_cols, target_col):
    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df.dropna()

    for col in model_df.columns:
        if model_df[col].dtype == "object":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))

    X = model_df[feature_cols]
    y = model_df[target_col]

    if len(model_df) < 10:
        return 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return round(model.score(X_test, y_test) * 100, 2)


if menu == "Patient Journey":
    st.header("🧑‍⚕️ Patient Journey / Hospital Admission Dataset")

    department = st.selectbox("Filter by Department", ["All"] + list(patient_df["Department"].unique()))
    search_patient = st.text_input("Search Patient ID")

    filtered_df = patient_df.copy()

    if department != "All":
        filtered_df = filtered_df[filtered_df["Department"] == department]

    if search_patient:
        filtered_df = filtered_df[
            filtered_df["Patient_ID"].astype(str).str.contains(search_patient, case=False)
        ]

    st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(filtered_df))
    col2.metric("Emergency Cases", filtered_df["Emergency_Flag"].astype(str).value_counts().get("True", 0))
    col3.metric("ICU Admissions", filtered_df["ICU_Admission"].astype(str).value_counts().get("True", 0))

    st.divider()

    st.subheader("Department Wise Patient Count")
    fig = px.bar(filtered_df["Department"].value_counts(), color_discrete_sequence=["#00E5FF"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Disease Type Distribution")
    fig = px.pie(filtered_df, names="Disease_Type", title="Disease Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML Model: Readmission Prediction")
    accuracy = ml_accuracy(
        patient_df,
        ["Age", "Waiting_Time", "Length_of_Stay"],
        "Readmission_Flag"
    )
    st.metric("Model Accuracy", f"{accuracy}%")


elif menu == "Bed Occupancy / ICU":
    st.header("🛏️ Bed Occupancy / ICU Dataset")

    ward = st.selectbox("Filter by Ward", ["All"] + list(bed_df["Ward"].unique()))
    filtered_df = bed_df.copy()

    if ward != "All":
        filtered_df = filtered_df[filtered_df["Ward"] == ward]

    st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Beds", int(filtered_df["Total_Beds"].sum()))
    col2.metric("Occupied Beds", int(filtered_df["Occupied_Beds"].sum()))
    col3.metric("Available Beds", int(filtered_df["Available_Beds"].sum()))

    st.divider()

    st.subheader("Emergency Load Distribution")
    fig = px.pie(filtered_df, names="Emergency_Load", title="Emergency Load Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ward Wise Records")
    fig = px.bar(filtered_df["Ward"].value_counts(), color_discrete_sequence=["#06D6A0"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML Model: Emergency Load Prediction")
    accuracy = ml_accuracy(
        bed_df,
        ["Total_Beds", "Occupied_Beds", "Available_Beds", "ICU_Beds", "ICU_Occupied", "Ventilator_Usage"],
        "Emergency_Load"
    )
    st.metric("Model Accuracy", f"{accuracy}%")


elif menu == "Medical Billing":
    st.header("💰 Medical Billing / Revenue Dataset")

    payment = st.selectbox("Filter by Payment Status", ["All"] + list(billing_df["Payment_Status"].unique()))
    search_patient = st.text_input("Search Patient ID")

    filtered_df = billing_df.copy()

    if payment != "All":
        filtered_df = filtered_df[filtered_df["Payment_Status"] == payment]

    if search_patient:
        filtered_df = filtered_df[
            filtered_df["Patient_ID"].astype(str).str.contains(search_patient, case=False)
        ]

    st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", int(filtered_df["Total_Bill"].sum()))
    col2.metric("Claim Amount", int(filtered_df["Claim_Amount"].sum()))
    col3.metric("Outstanding Amount", int(filtered_df["Outstanding_Amount"].sum()))

    st.divider()

    st.subheader("Payment Status Distribution")
    fig = px.pie(filtered_df, names="Payment_Status", title="Payment Status Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insurance Provider Wise Revenue")
    fig = px.bar(
        filtered_df,
        x="Insurance_Provider",
        y="Total_Bill",
        color="Insurance_Provider",
        title="Insurance Provider Wise Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML Model: Claim Status Prediction")
    accuracy = ml_accuracy(
        billing_df,
        ["Treatment_Cost", "Medicine_Cost", "Room_Charges", "Lab_Charges", "Total_Bill", "Claim_Amount"],
        "Claim_Status"
    )
    st.metric("Model Accuracy", f"{accuracy}%")


elif menu == "Emergency Room":
    st.header("🚑 Emergency Room Dataset")

    severity = st.selectbox("Filter by Severity Level", ["All"] + list(er_df["Severity_Level"].unique()))
    filtered_df = er_df.copy()

    if severity != "All":
        filtered_df = filtered_df[filtered_df["Severity_Level"] == severity]

    st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total ER Cases", len(filtered_df))
    col2.metric("Average Wait Time", round(filtered_df["Wait_Time"].mean(), 2))
    col3.metric("Average Treatment Time", round(filtered_df["Treatment_Time"].mean(), 2))

    st.divider()

    st.subheader("Severity Level Distribution")
    fig = px.pie(filtered_df, names="Severity_Level", title="Severity Level Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Triage Level Count")
    fig = px.bar(filtered_df["Triage_Level"].value_counts(), color_discrete_sequence=["#8338EC"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML Model: Admission or Discharge Prediction")
    accuracy = ml_accuracy(
        er_df,
        ["Wait_Time", "Treatment_Time"],
        "Admission_or_Discharge"
    )
    st.metric("Model Accuracy", f"{accuracy}%")


elif menu == "Staff Allocation":
    st.header("👨‍⚕️ Staff Allocation Dataset")

    role = st.selectbox("Filter by Staff Role", ["All"] + list(staff_df["Staff_Role"].unique()))
    filtered_df = staff_df.copy()

    if role != "All":
        filtered_df = filtered_df[filtered_df["Staff_Role"] == role]

    st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Staff Records", len(filtered_df))
    col2.metric("Patients Assigned", int(filtered_df["Patients_Assigned"].sum()))
    col3.metric("Overtime Hours", int(filtered_df["Overtime_Hours"].sum()))

    st.divider()

    st.subheader("Staff Role Distribution")
    fig = px.pie(filtered_df, names="Staff_Role", title="Staff Role Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Department Wise Staff Count")
    fig = px.bar(filtered_df["Department"].value_counts(), color_discrete_sequence=["#118AB2"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML Model: Availability Prediction")
    accuracy = ml_accuracy(
        staff_df,
        ["Patients_Assigned", "Overtime_Hours"],
        "Availability"
    )
    st.metric("Model Accuracy", f"{accuracy}%")


elif menu == "Operation Theatre":
    st.header("🏥 Operation Theatre / OT Dataset")

    surgery_type = st.selectbox("Filter by Surgery Type", ["All"] + list(ot_df["Surgery_Type"].unique()))
    filtered_df = ot_df.copy()

    if surgery_type != "All":
        filtered_df = filtered_df[filtered_df["Surgery_Type"] == surgery_type]

    st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Surgeries", len(filtered_df))
    col2.metric("Average Delay", round(filtered_df["Delay_Time"].mean(), 2))
    col3.metric("Maximum Delay", int(filtered_df["Delay_Time"].max()))

    st.divider()

    st.subheader("Surgery Type Distribution")
    fig = px.pie(filtered_df, names="Surgery_Type", title="Surgery Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("OT Room Usage")
    fig = px.bar(filtered_df["OT_Room"].value_counts(), color_discrete_sequence=["#06D6A0"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Equipment Usage Count")
    fig = px.bar(filtered_df["Equipment_Used"].value_counts(), color_discrete_sequence=["#FFD166"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML Model: Delay Category Prediction")

    temp_df = ot_df.copy()
    temp_df["Delay_Category"] = temp_df["Delay_Time"].apply(
        lambda x: "High Delay" if x > 60 else "Low Delay"
    )

    accuracy = ml_accuracy(
        temp_df,
        ["Delay_Time"],
        "Delay_Category"
    )
    st.metric("Model Accuracy", f"{accuracy}%")