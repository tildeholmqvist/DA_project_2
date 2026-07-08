import streamlit as st
from app_pages.multi_page import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_prediction import page_prediction_body
from app_pages.page_findings import page_findings_body

app = MultiPage(app_name="Car Price Predictor 🚗")

app.add_page("Project Summary", page_summary_body)
app.add_page("Car Price Prediction", page_prediction_body)
app.add_page("Key Findings", page_findings_body)

app.run()