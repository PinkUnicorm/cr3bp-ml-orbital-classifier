"""
app.py

Interactive Streamlit demo for the CR3BP orbital stability classifier.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from cr3bp import make_features, integrate_orbit


st.set_page_config(
    page_title="CR3BP Orbital Stability Classifier",
    layout="wide"
)

st.title("Circular Restricted 3-Body Problem Orbital Stability Classifier")

st.markdown(
    """
    This app predicts whether a small object, such as a spacecraft or asteroid,
    will remain bounded, escape, or collide based on its initial conditions.

    In the Circular Restricted Three-Body Problem (CR3BP), the two large bodies
    are defined by their mass ratio, while the third body is treated as a tiny
    test object. The third object's mass is shown for clarity, but it does not
    affect the physics calculation in this simplified model.
    """
)

# Load trained model and scaler
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

label_names = {
    0: "Stable / Bounded",
    1: "Escape",
    2: "Collision"
}

preset_masses = {
    "Earth-Moon example": {
        "primary_label": "Earth",
        "secondary_label": "Moon",
        "third_label": "Spacecraft",
        "m1": 5.972e24,
        "m2": 7.348e22,
        "m3": 1000.0,
    },
    "Sun-Earth example": {
        "primary_label": "Sun",
        "secondary_label": "Earth",
        "third_label": "Spacecraft",
        "m1": 1.989e30,
        "m2": 5.972e24,
        "m3": 1000.0,
    },
    "Custom masses": {
        "primary_label": "Primary body",
        "secondary_label": "Secondary body",
        "third_label": "Small object",
        "m1": 5.972e24,
        "m2": 7.348e22,
        "m3": 1000.0,
    },
}

st.sidebar.header("System Setup")

preset_name = st.sidebar.selectbox(
    "Mass preset",
    options=list(preset_masses.keys()),
    help="Choose a real-world-style pair of large bodies or enter custom masses."
)

preset = preset_masses[preset_name]

st.sidebar.markdown("### Masses")
st.sidebar.caption(
    "m1 and m2 define the two large bodies. m3 is included for understanding, "
    "but CR3BP assumes the third body is too small to affect the two large bodies."
)

m1 = st.sidebar.number_input(
    f"m1: {preset['primary_label']} mass (kg)",
    min_value=1.0,
    value=float(preset["m1"]),
    format="%.6e",
    help="Mass of the larger primary body."
)

m2 = st.sidebar.number_input(
    f"m2: {preset['secondary_label']} mass (kg)",
    min_value=1.0,
    value=float(preset["m2"]),
    format="%.6e",
    help="Mass of the smaller secondary body."
)

m3 = st.sidebar.number_input(
    f"m3: {preset['third_label']} mass (kg)",
    min_value=0.0,
    value=float(preset["m3"]),
    format="%.6e",
    help="Shown for clarity only. In CR3BP, this mass does not affect the path."
)

# CR3BP convention requires the smaller mass parameter to be in (0, 0.5].
# If the user enters m1 < m2, this formula still works by using the smaller
# body as m2 for the normalized CR3BP mass ratio.
larger_mass = max(m1, m2)
smaller_mass = min(m1, m2)
mu = smaller_mass / (larger_mass + smaller_mass)

st.sidebar.info(f"Calculated mass ratio μ = {mu:.8f}")

st.sidebar.markdown("### Initial Conditions for the Third Body")
st.sidebar.caption(
    "These sliders control where the small object starts and how fast it is moving."
)

x0 = st.sidebar.slider(
    "Initial x-position",
    min_value=-1.5,
    max_value=1.5,
    value=0.5,
    step=0.01,
    help="Normalized x-position of the third body."
)

y0 = st.sidebar.slider(
    "Initial y-position",
    min_value=-1.5,
    max_value=1.5,
    value=0.0,
    step=0.01,
    help="Normalized y-position of the third body."
)

vx0 = st.sidebar.slider(
    "Initial x-velocity",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    help="Normalized x-velocity of the third body."
)

vy0 = st.sidebar.slider(
    "Initial y-velocity",
    min_value=-1.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Normalized y-velocity of the third body."
)

# Create feature vector
features = make_features(x0, y0, vx0, vy0, mu)
features_scaled = scaler.transform([features])

prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Prediction")
    st.success(f"Predicted outcome: **{label_names[prediction]}**")

    probability_df = pd.DataFrame({
        "Outcome": [label_names[label] for label in model.classes_],
        "Probability": probabilities
    })

    st.subheader("Prediction Confidence")
    st.bar_chart(
        probability_df,
        x="Outcome",
        y="Probability"
    )

with col2:
    st.subheader("Selected System and Initial Condition")

    st.write(f"**Primary body mass:** {m1:.3e} kg")
    st.write(f"**Secondary body mass:** {m2:.3e} kg")
    st.write(f"**Third body mass:** {m3:.3e} kg *(not used in CR3BP dynamics)*")
    st.write(f"**Calculated mass ratio μ:** {mu:.8f}")
    st.write(f"**Initial position:** ({x0:.2f}, {y0:.2f})")
    st.write(f"**Initial velocity:** ({vx0:.2f}, {vy0:.2f})")
    st.write(f"**Jacobi constant:** {features[5]:.4f}")

    with st.expander("What does μ mean?"):
        st.markdown(
            """
            μ is the normalized mass ratio between the two large bodies:

            **μ = smaller mass / (larger mass + smaller mass)**

            For the Earth-Moon system, μ is about **0.01215**. The third body's
            mass is ignored because this app uses the **restricted** three-body
            model, where the third body is assumed to be too small to pull on the
            two large bodies.
            """
        )

st.divider()

st.subheader("Numerically Integrated 2D Trajectory")

initial_state = [x0, y0, vx0, vy0]

solution = integrate_orbit(
    initial_state=initial_state,
    mu=mu,
    t_max=20.0,
    n_points=1000
)

x = solution.y[0]
y = solution.y[1]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, linewidth=1.5, label="Third body trajectory")
ax.scatter(x0, y0, s=60, label="Third body start")

# Primary body locations in normalized CR3BP coordinates.
# The larger primary body is placed at -mu and the smaller secondary body at 1 - mu.
primary_1_x = -mu
primary_2_x = 1 - mu

ax.scatter(primary_1_x, 0, s=140, marker="o", label="Large body 1")
ax.scatter(primary_2_x, 0, s=90, marker="o", label="Large body 2")

ax.set_xlabel("Normalized x-position")
ax.set_ylabel("Normalized y-position")
ax.set_title("Simulated Trajectory of the Third Body")
ax.grid(True)
ax.axis("equal")
ax.legend()

st.pyplot(fig)

st.caption(
    "Note: The plot uses normalized CR3BP coordinates. The distance between the two large bodies is scaled to 1, "
    "so the axes are not kilometers, meters, or seconds."
)
