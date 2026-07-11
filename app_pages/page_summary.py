import streamlit as st


def page_summary_body():
    """
    This function displays the Project Summary page.
    """
    st.write("# Car Price Analysis — Project Summary")
    st.write("---")

    st.info(
        "This project analyses car price data to identify key pricing "
        "factors and predict car prices using machine learning."
    )

    st.write("## Dataset")
    st.write(
        "* 205 cars with 26 features from the US automobile market, "
        "across 22 brands"
    )
    st.write("* Source: Kaggle — Car Price Prediction dataset")

    st.write("## Business Hypotheses")
    st.write(
        "1. Engine size and horsepower are the strongest predictors "
        "of price"
    )
    st.write(
        "2. Luxury brands have significantly higher prices than "
        "economy brands"
    )
    st.write(
        "3. Diesel cars have significantly higher prices than "
        "petrol cars"
    )
    st.write("4. Car body style significantly influences price")

    st.write("## Key Results")
    st.write("* Best model: ExtraTreesRegressor (R²=0.911 on test set)")
    st.write("* Strongest price predictor: Engine size (importance: 0.20)")