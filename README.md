README.md Content
🏠 NoBrokerage AI Property Search (Streamlit + FastAPI)
An AI-powered application that uses Natural Language Processing (NLP) to convert conversational property search queries into structured filters, displaying matching real estate listings from a backend service (assumed to be FastAPI).

✨ Features
Natural Language Search: Users can query properties using conversational phrases (e.g., "3BHK flat in Pune under ₹1.2 Cr").

Intelligent Filtering: The AI backend extracts constraints (city, BHK, price range, possession status, amenities) from the text.

Chat Interface: Presents results in a clean chat-style interface with a summary of applied filters.

Property Cards: Displays matching results in responsive, visually appealing property cards using custom HTML/CSS.

API Configuration: Easy configuration for the backend FastAPI URL via the Streamlit sidebar.

🛠️ Project Structure (Conceptual)
This application requires two main components running simultaneously:

Frontend: The Streamlit application (streamlit_app.py - the code provided).

Backend: A FastAPI server (not provided) that handles the NLP query parsing, database search, and returns the structured JSON data.

🚀 Getting Started
Prerequisites
You must have Python installed. It is highly recommended to use a virtual environment.

Bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate   # On Windows
1. Backend Setup (Placeholder)
Note: This application requires a backend API running to function correctly. The frontend is configured to hit http://localhost:8000 by default.

Start your FastAPI/Backend service on the configured port (default: 8000).

Bash

# Example command to run a typical FastAPI app:
uvicorn main:app --reload --port 8000
The backend must expose endpoints like /health, /cities, and /api/query as expected by the Streamlit app.

2. Frontend Setup
Install Dependencies:

Bash

pip install streamlit requests
Save the Code: Save the provided Python code as streamlit_app.py.

Run the Streamlit App:

Bash

streamlit run streamlit_app.py
The application will open in your default browser at http://localhost:8501.

💻 Usage
Configure API URL: If your backend is running on a different URL or port, update the Backend API URL field in the Settings (Sidebar). Click Test Connection to verify connectivity.

Enter Query: Use the input box at the bottom of the screen.

Search: Click the 🚀 Search button or press Enter.

Example Queries:
3BHK flat in Pune under 1.2 Cr

Ready to move 2BHK in Mumbai

Properties in Bangalore under 80 lakhs with a balcony

I'm looking for an under construction property in Pune for sale

The AI will respond with a Summary Box detailing the applied filters and a display of matching Property Cards.