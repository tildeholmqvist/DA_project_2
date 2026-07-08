import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv('outputs/datasets/cleaned/car_prices_cleaned.csv')
    df['CarBrand'] = df['CarName'].str.split().str[0].str.lower()
    df['CarBrand'] = df['CarBrand'].replace({
        'maxda': 'mazda',
        'porcshce': 'porsche',
        'toyouta': 'toyota',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen'
    })
    return df

def page_findings_body():
    df = load_data()

    st.write("# Key Findings")
    st.write("---")
    st.info(
        "This page presents the key findings from the EDA analysis, "
        "including hypothesis testing results and visualisations."
    )

    # Hypothesis 1
    st.write("## Hypothesis 1 — Engine Size and Horsepower vs Price")
    st.write(
        "H0: Engine size and horsepower have no significant correlation with car price.\n\n"
        "H1: Engine size and horsepower are significantly correlated with car price."
    )
    fig1a = px.scatter(df, x='enginesize', y='price',
                       title='Engine Size vs Price',
                       labels={'enginesize': 'Engine Size', 'price': 'Price (USD)'},
                       hover_data=['CarBrand'],
                       trendline='ols')
    st.plotly_chart(fig1a)

    fig1b = px.scatter(df, x='horsepower', y='price',
                       title='Horsepower vs Price',
                       labels={'horsepower': 'Horsepower', 'price': 'Price (USD)'},
                       hover_data=['CarBrand'],
                       trendline='ols')
    st.plotly_chart(fig1b)

    num_cols = ['enginesize', 'horsepower', 'curbweight', 'carlength',
                'carwidth', 'wheelbase', 'citympg', 'highwaympg', 'boreratio', 'stroke']
    price_corr = df[num_cols].corrwith(df['price']).sort_values()

    fig1c = px.bar(x=price_corr.values, y=price_corr.index, orientation='h',
                   title='Correlation of All Features with Price',
                   labels={'x': 'Correlation with Price', 'y': 'Feature'})
    st.plotly_chart(fig1c)

    st.success("✅ Hypothesis 1 partially confirmed — Engine size (0.87) is the "
               "strongest predictor, but curbweight (0.84) unexpectedly outperformed "
               "horsepower (0.81)")

    st.write("---")

    # Hypothesis 2
    st.write("## Hypothesis 2 — Luxury vs Economy Brands")
    st.write(
        "H0: There is no significant price difference between luxury and economy brands.\n\n"
        "H1: Luxury car brands have significantly higher prices than economy brands."
    )
    avg_price_brand = df.groupby('CarBrand')['price'].mean().sort_values(ascending=False).reset_index()
    avg_price_brand['brand_type'] = avg_price_brand['price'].apply(
        lambda x: 'Luxury' if x > 20000 else 'Economy')
    fig2 = px.bar(avg_price_brand, x='CarBrand', y='price',
                  color='brand_type',
                  title='Average Price by Car Brand (Luxury vs Economy)',
                  labels={'CarBrand': 'Car Brand', 'price': 'Average Price (USD)'},
                  color_discrete_map={'Luxury': 'darkblue', 'Economy': 'lightblue'})
    st.plotly_chart(fig2)
    st.success("✅ Hypothesis 2 confirmed — Luxury brands are significantly more "
               "expensive (p-value = 0.0000)")

    st.write("---")

    # Hypothesis 3
    st.write("## Hypothesis 3 — Diesel vs Petrol Cars")
    st.write(
        "H0: There is no significant price difference between diesel and petrol cars.\n\n"
        "H1: Diesel cars have significantly higher prices than petrol cars."
    )
    st.warning("⚠️ Dataset imbalance: 185 gas cars vs 20 diesel cars — "
               "results should be interpreted with caution.")

    avg_price_fuel = df.groupby('fueltype')['price'].mean().reset_index()
    fig3a = px.bar(avg_price_fuel, x='fueltype', y='price',
                   title='Average Price by Fuel Type (Imbalanced — 185 vs 20)',
                   labels={'fueltype': 'Fuel Type', 'price': 'Average Price (USD)'},
                   color='fueltype',
                   color_discrete_map={'gas': 'lightblue', 'diesel': 'darkblue'})
    st.plotly_chart(fig3a)

    gas_sample = df[df['fueltype'] == 'gas']['price'].sample(n=20, random_state=42)
    diesel_sample = df[df['fueltype'] == 'diesel']['price']
    balanced_df = pd.DataFrame({
        'price': pd.concat([diesel_sample, gas_sample]),
        'fueltype': ['diesel'] * 20 + ['gas'] * 20
    })
    avg_balanced = balanced_df.groupby('fueltype')['price'].mean().reset_index()

    fig3b = px.bar(avg_balanced, x='fueltype', y='price',
                   title='Average Price by Fuel Type (Balanced Sample — 20 vs 20)',
                   labels={'fueltype': 'Fuel Type', 'price': 'Average Price (USD)'},
                   color='fueltype',
                   color_discrete_map={'gas': 'lightblue', 'diesel': 'darkblue'})
    st.plotly_chart(fig3b)

    st.error("❌ Hypothesis 3 rejected — With balanced sampling, no significant "
             "price difference between diesel and gas cars (p-value = 0.8659)")

    st.write("---")

    # Hypothesis 4
    st.write("## Hypothesis 4 — Body Style vs Price")
    st.write(
        "H0: Car body style has no significant influence on price.\n\n"
        "H1: Car body style significantly influences car price."
    )
    fig4 = px.box(df, x='carbody', y='price',
                  title='Price Distribution by Car Body Style',
                  labels={'carbody': 'Body Style', 'price': 'Price (USD)'},
                  color='carbody')
    st.plotly_chart(fig4)
    st.success("✅ Hypothesis 4 confirmed — Body style significantly influences "
               "price (p-value = 0.0000). Convertibles are the most expensive.")

    st.write("---")

    # Feature Importance
    st.write("## Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['enginesize', 'curbweight', 'cylindernumber', 'carwidth',
                    'horsepower', 'citympg', 'drivewheel', 'highwaympg'],
        'Importance': [0.20, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.07]
    }).sort_values('Importance', ascending=True)
    fig5 = px.bar(importance_df, x='Importance', y='Feature',
                  orientation='h',
                  title='Feature Importance — ExtraTreesRegressor')
    st.plotly_chart(fig5)

    st.write("---")

    # Model Performance
    st.write("## Model Performance")
    st.write("* **Best model:** ExtraTreesRegressor")
    st.write("* **Test R²:** 0.911")
    st.write("* **Test RMSE:** $2,619")
    st.write("* **Test MAE:** $1,572")

    st.write("---")

    # Limitations
    st.write("## Limitations")
    st.warning(
        "* Small dataset (205 rows) — results should be interpreted with caution\n"
        "* Imbalanced fuel type (185 gas vs 20 diesel) — explains why H3 was rejected\n"
        "* Mild overfitting (train R²=0.998 vs test R²=0.911)\n"
        "* T-test for H2 was run on brand averages (n=22) rather than individual cars"
    )

    st.write("---")

    # Business Recommendations
    st.write("## Business Recommendations")
    st.success(
        "* Manufacturers targeting the US luxury segment should prioritize "
        "engine size and body style (convertibles) in pricing strategy\n"
        "* European brands command a significant price premium\n"
        "* Diesel engines are not a significant price differentiator in the US market"
    )