import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model_and_encoder():
    model = joblib.load('outputs/models/car_price_model.pkl')
    encoder = joblib.load('outputs/models/encoder.pkl')
    return model, encoder


def page_prediction_body():
    """
    This function displays the Car Price Prediction page.
    """
    st.write("# Enter Car Specifications to Predict Price")
    st.write("---")

    st.info(
        "Enter the car's specifications below to predict its price "
        "using our trained ExtraTreesRegressor model (Test R² = 0.911)."
    )

    model, encoder = load_model_and_encoder()

    st.write("## Engine & Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        enginesize = st.number_input(
            "Engine size", min_value=61, max_value=326, value=120
        )
        horsepower = st.number_input(
            "Horsepower", min_value=48, max_value=288, value=100
        )
    with col2:
        curbweight = st.number_input(
            "Curb weight", min_value=1488, max_value=4066, value=2400
        )
        cylindernumber = st.selectbox(
            "Cylinder number", [2, 3, 4, 5, 6, 8, 12], index=2
        )
    with col3:
        citympg = st.number_input(
            "City MPG", min_value=13, max_value=49, value=25
        )
        highwaympg = st.number_input(
            "Highway MPG", min_value=16, max_value=54, value=30
        )

    st.write("## Body & Dimensions")
    col4, col5, col6 = st.columns(3)
    with col4:
        carbody = st.selectbox(
            "Body style",
            ['convertible', 'hardtop', 'hatchback', 'sedan', 'wagon']
        )
        carwidth = st.number_input(
            "Car width", min_value=60.3, max_value=72.3, value=65.0
        )
    with col5:
        carlength = st.number_input(
            "Car length", min_value=141.1, max_value=208.1, value=175.0
        )
        doornumber = st.selectbox("Door number", [2, 4], index=1)
    with col6:
        drivewheel = st.selectbox("Drive wheel", ['fwd', 'rwd', '4wd'])
        wheelbase = st.number_input(
            "Wheelbase", min_value=86.6, max_value=120.9, value=98.0
        )

    st.write("## Brand & Other Specs")
    col7, col8, col9 = st.columns(3)
    brand_list = [
        'alfa-romero', 'audi', 'bmw', 'buick', 'chevrolet', 'dodge',
        'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi',
        'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab',
        'subaru', 'toyota', 'volkswagen', 'volvo'
    ]
    with col7:
        CarBrand = st.selectbox("Car brand", brand_list)
        fueltype = st.selectbox("Fuel type", ['gas', 'diesel'])
    with col8:
        aspiration = st.selectbox("Aspiration", ['std', 'turbo'])
        enginelocation = st.selectbox("Engine location", ['front', 'rear'])
    with col9:
        enginetype = st.selectbox(
            "Engine type",
            ['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor']
        )
        fuelsystem = st.selectbox(
            "Fuel system",
            ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi']
        )

    st.write("## Additional Specs")
    col10, col11 = st.columns(2)
    with col10:
        boreratio = st.number_input(
            "Bore ratio", min_value=2.54, max_value=3.94, value=3.3
        )
        stroke = st.number_input(
            "Stroke", min_value=2.07, max_value=4.17, value=3.3
        )
    with col11:
        compressionratio = st.number_input(
            "Compression ratio", min_value=7.0, max_value=23.0, value=9.0
        )
        peakrpm = st.number_input(
            "Peak RPM", min_value=4150, max_value=6600, value=5200
        )

    st.write("---")

    if st.button("Predict Price"):
        input_data = pd.DataFrame([{
            'enginesize': enginesize,
            'horsepower': horsepower,
            'curbweight': curbweight,
            'carwidth': carwidth,
            'carlength': carlength,
            'wheelbase': wheelbase,
            'boreratio': boreratio,
            'stroke': stroke,
            'compressionratio': compressionratio,
            'peakrpm': peakrpm,
            'citympg': citympg,
            'highwaympg': highwaympg,
            'fueltype': fueltype,
            'aspiration': aspiration,
            'doornumber': doornumber,
            'carbody': carbody,
            'drivewheel': drivewheel,
            'enginelocation': enginelocation,
            'enginetype': enginetype,
            'cylindernumber': cylindernumber,
            'fuelsystem': fuelsystem,
            'CarBrand': CarBrand
        }])

        categorical_cols = [
            'fueltype', 'aspiration', 'carbody', 'drivewheel',
            'enginelocation', 'enginetype', 'fuelsystem', 'CarBrand'
        ]
        for col in categorical_cols:
            input_data[col] = input_data[col].astype(str)

        input_encoded = encoder.transform(input_data)
        prediction = model.predict(input_encoded)[0]

        st.success(f"## Predicted Price: ${prediction:,.2f}")
        st.write(
            "This prediction has an average error of ±$1,572 "
            "(based on the model's test set MAE)."
        )
