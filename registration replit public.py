from groq import Groq
import streamlit as st
import os
from crewai import Crew, Agent, Task, Process
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from pyairtable import Table
import datetime

AIRTABLE_API_KEY = st.secrets["AIRTABLE_API_KEY"]
BASE_ID = st.secrets["BASE_ID"]
TABLE_NAME = st.secrets["TABLE_NAME"]  # Replace with your table name
groq_api_key = st.secrets["GROQ_API_KEY"]

# set environment variable for GROQ API key

os.environ["GROQ_API_KEY"] = groq_api_key

airtable = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)


# Initialize session state to keep track of the process
if 'goals_processed' not in st.session_state:
    st.session_state.goals_processed = False
    st.session_state.goals = ""


# add logo of idealab on top left of te page

st.image("Screenshot (1497).png", width=600)

# Create title for WHU MBA Streamlit App
st.title("Registration Buddy Matching")

st.write("Welcome to the registration form for the Buddy Matching during Idealab. The information you provide will be used to identify some interesting people, who also attend the Idealab conference and who can help you in realizing your networking goals. At the beginning of Idealab, we will send you a document that will contain the names of the people with whom you are matched.") 
         
st.markdown("**Please execute the following steps**")
st.write("Step 1: Fill in the form below to register for the event and upload your LinkedIn profile.")
st.write("Step 2: Based on your LinkedIn profile, the application will give you a first suggestion of networking goals that could be relevant for you during Idealab. You can review and adjust these suggestions to your preference.")
st.write("Step 3: After you have adjusted the networking goals, you need to click on Save Adjusted Networking Goals. This will end the registration process")

text = ""

# Display initial input form for user details and PDF upload
with st.form("registration_form"):
    name = st.text_input("Please enter your first name and last name")
    email = st.text_input("Please enter your email address (we will send the document with matched buddies to this email address)")
    # user needs to indicate with yes or no if they are a student
    uploaded_file = st.file_uploader("Please upload a PDF of your LinkedIn profile. You can find this PDF by going to your LinkedIn profile page, click on More, and click on Save PDF. By uploading the file, you agree that we use and store your LinkedIn profile for the purpose of matching durng IdeaLab.", type="pdf")
    submit_form = st.form_submit_button("Submit")

# If the form is submitted
if submit_form and not st.session_state.goals_processed:
    if uploaded_file is not None:
        # Read the pdf file
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.session_state.profile = text

        client = Groq()
        GROQ_LLM = ChatGroq(model="llama3-70b-8192")

        # Create agent to analyze the text and identify core skills
        goal_identifier = Agent(
            role='Identify core networking objectives based on the LinkedIn profile',
            goal=f"""Identify core networking objectives based on the LinkedIn profile.""", 
            backstory=f"""You are a great expert in helping people to identify which goals and objectives they should pursue during networking events. 
            You have been trained to rely on the LinkedIn profile of persosn to formulate specific networking goals during 
            Idealab, which is a networking event for students, founders and investors who share interest in the topic of entrepreneurship""",  
            verbose=True,
            llm=GROQ_LLM,
            allow_delegation=False,
            max_iter=5,
        )

        # Create a task to identify the core skills
        identify_goals = Task(
            description=f""" Based on the LinkedIn profile provided below, identify and suggest three specific networking goals that the person should pursue during the upcoming Idealab conference. 
                        Rules:The networking goals should be: (i) Tailored to their professional background, skills, interests, and career objectives; (ii) Actionable and specific, enabling the person to make meaningful connections. (iii)Aligned with the themes and opportunities typically available at the Idealab conference.
                        Instructions: Begin by briefly summarizing the person's current professional status and objectives based on their LinkedIn profile. For each networking goal, provide a clear and concise description. Present the goals in a numbered list for clarity.
                        Here is the LInkedIn profile: {st.session_state.profile} """,
            expected_output='As output, provide a clear description of the identified networking objectives.',
            agent=goal_identifier,
        )

        crew = Crew(
            agents=[goal_identifier],
            tasks=[identify_goals],
            process=Process.sequential,
            share_crew=False,
        )
        results = crew.kickoff()
        st.session_state.goals = identify_goals.output.exported_output
        st.session_state.goals_processed = True

        # Display identified skills in a text area for user adjustment

        with st.form("adjust_goals_form"):
            adjusted_goals = st.text_area("Please review and adjust the networking goals below:", value=st.session_state.goals)
            save_adjusted_goals = st.form_submit_button("Save Adjusted Networking Goals")

        if save_adjusted_goals:
            st.session_state.goals = adjusted_goals
            st.write("Adjusted Networking Goal Saved:")
            st.write(st.session_state.goals)
            st.markdown("**Registration successful. You can now leave the registration.**")
    else:
        st.error("Please upload a PDF file.")

# If skills have already been processed, allow adjustment without re-running the agents
elif st.session_state.goals_processed:
    with st.form("adjust_goals_form"):
        adjusted_goals = st.text_area("Please review and adjust the identified goals below:", value=st.session_state.goals)
        save_adjusted_goals = st.form_submit_button("Save Adjusted Networking Goals")

    if save_adjusted_goals:
        st.session_state.goals = adjusted_goals
        st.write("Adjusted Goals Saved:")
        st.write(st.session_state.goals)
        st.markdown("**Registration successful. You can now leave the registration.**")
        record = {"Name": name, "Email": email, "LinkedIn Profile": st.session_state.profile, "Goals": st.session_state.goals, "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        airtable.create(record)

