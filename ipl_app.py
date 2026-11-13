import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# LOAD MODEL & ENCODER
model = pickle.load(open('ipl_model.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# TEAM LIST 
teams = list(le.classes_)

# UI
st.title("🏏 IPL Win Probability Predictor")

st.write("### Enter Match Situation")

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)

# Prevent same team
if batting_team == bowling_team:
    st.error("Batting and Bowling team cannot be same")
    st.stop()

target = st.slider("Target Score", 100, 250, 180)
current_score = st.slider("Current Score", 0, target, 100)
overs = st.slider("Overs Completed", 0.1, 20.0, 10.0)
wickets = st.slider("Wickets Lost", 0, 10, 2)

# PREDICTION
if st.button("Predict"):

    # balls played calculation
    balls_played = int(overs)*6 + int((overs-int(overs))*10)

    runs_left = target - current_score
    balls_left = 120 - balls_played
    wickets_left = 10 - wickets

    # safe calculations
    crr = current_score / overs if overs > 0 else 0
    rrr = runs_left / (balls_left/6) if balls_left > 0 else 0

    # encode teams
    batting_team_enc = le.transform([batting_team])[0]
    bowling_team_enc = le.transform([bowling_team])[0]

    # input dataframe
    input_df = pd.DataFrame([[
        batting_team_enc,
        bowling_team_enc,
        runs_left,
        balls_left,
        wickets_left,
        crr,
        rrr
    ]], columns=[
        'batting_team','bowling_team','runs_left','balls_left',
        'wickets_left','crr','rrr'
    ])

    # prediction
    prob = model.predict_proba(input_df)[0]

    win_prob = prob[1]*100
    lose_prob = prob[0]*100

    # OUTPUT
    st.subheader("📊 Win Probability")

    col1, col2 = st.columns(2)

    col1.metric(label=batting_team, value=f"{round(win_prob,2)}%")
    col2.metric(label=bowling_team, value=f"{round(lose_prob,2)}%")

    # match situation
    if win_prob > 50:
        st.success(f"Match Situation: Favoring {batting_team}")
    else:
        st.warning(f"Match Situation: Favoring {bowling_team}")

    st.write(f"📈 Required Run Rate: {round(rrr,2)}")
    st.write(f"📉 Current Run Rate: {round(crr,2)}")

# REAL PROBABILITY CURVE
st.subheader("📉 Real Win Probability Curve")

overs_list = []
prob_list = []

temp_score = 0
temp_wickets = 0

# simulate match progression
for o in np.arange(0.5, overs + 0.1, 0.5):

    balls = int(o)*6 + int((o-int(o))*10)

    # simulate score progression (approx)
    temp_score = (current_score / overs) * o
    temp_wickets = int((wickets / overs) * o)

    runs_left_sim = target - temp_score
    balls_left_sim = 120 - balls
    wickets_left_sim = 10 - temp_wickets

    crr_sim = temp_score / o if o > 0 else 0
    rrr_sim = runs_left_sim / (balls_left_sim/6) if balls_left_sim > 0 else 0

    # encode
    batting_team_enc = le.transform([batting_team])[0]
    bowling_team_enc = le.transform([bowling_team])[0]

    input_sim = pd.DataFrame([[
        batting_team_enc,
        bowling_team_enc,
        runs_left_sim,
        balls_left_sim,
        wickets_left_sim,
        crr_sim,
        rrr_sim
    ]], columns=[
        'batting_team','bowling_team','runs_left','balls_left',
        'wickets_left','crr','rrr'
    ])

    prob_sim = model.predict_proba(input_sim)[0][1] * 100

    overs_list.append(o)
    prob_list.append(prob_sim)

# plot
fig, ax = plt.subplots()
ax.plot(overs_list, prob_list)
ax.set_xlabel("Overs")
ax.set_ylabel("Win Probability (%)")
ax.set_title(f"{batting_team} Win Probability Over Time")

st.pyplot(fig)

# INFO
st.info("This prediction is based on match conditions like runs required, wickets left, and overs completed.")