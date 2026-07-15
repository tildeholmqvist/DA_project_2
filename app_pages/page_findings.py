import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def load_data():
    """
    Loads the cleaned car price dataset and extracts CarBrand from
    CarName, applying the same typo corrections used in the ML
    notebook (03_ML.ipynb) to keep results consistent.
    """
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
    """
    Displays the Key Findings page, presenting the EDA and ML results
    organised by hypothesis (H1-H4), followed by feature importance,
    model performance, limitations, and business recommendations.
    """
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
        "H0: Engine size and horsepower have no significant "
        "correlation with car price.\n\n"
        "H1: Engine size and horsepower are significantly correlated "
        "with car price."
    )
    fig1a = px.scatter(
        df, x='enginesize', y='price',
        title='Engine Size vs Price',
        labels={'enginesize': 'Engine Size', 'price': 'Price (USD)'},
        hover_data=['CarBrand'],
        trendline='ols'
    )
    st.plotly_chart(fig1a)
    st.caption(
        "Each dot on this chart is one car. The further right a dot "
        "is, the bigger that car's engine; the higher up it is, the "
        "more expensive it is. The line running through the dots is "
        "a **trendline** — if it slopes upward, that tells us bigger "
        "engines tend to cost more. This tendency for two variables "
        "to move together is called **correlation** — a core "
        "statistical concept used throughout this analysis."
    )

    fig1b = px.scatter(
        df, x='horsepower', y='price',
        title='Horsepower vs Price',
        labels={'horsepower': 'Horsepower', 'price': 'Price (USD)'},
        hover_data=['CarBrand'],
        trendline='ols'
    )
    st.plotly_chart(fig1b)
    st.caption(
        "This chart works the same way, but compares horsepower "
        "(engine power output) to price instead of engine size. The "
        "upward-sloping trendline shows the same kind of pattern: "
        "more powerful cars tend to cost more."
    )

    num_cols = [
        'enginesize', 'horsepower', 'curbweight', 'carlength',
        'carwidth', 'wheelbase', 'citympg', 'highwaympg',
        'boreratio', 'stroke'
    ]
    price_corr = df[num_cols].corrwith(df['price']).sort_values()

    fig1c = px.bar(
        x=price_corr.values, y=price_corr.index, orientation='h',
        title='Correlation of All Features with Price',
        labels={'x': 'Correlation with Price', 'y': 'Feature'}
    )
    st.plotly_chart(fig1c)
    st.caption(
        "This chart ranks every numeric feature by its **correlation** "
        "with price — a score from -1 to +1 showing how strongly two "
        "things move together. A score close to +1 (like enginesize "
        "at 0.87) means a strong tendency to rise together; a score "
        "close to 0 means little to no relationship. The longer the "
        "bar reaches to the right, the stronger the connection to "
        "price."
    )

    st.success(
        "✅ Answering the question 'Do engine size and horsepower "
        "significantly correlate with price?' — Hypothesis 1 "
        "partially confirmed: Engine size (0.87) is the strongest "
        "predictor, but curbweight (0.84) unexpectedly outperformed "
        "horsepower (0.81)"
    )

    st.write("---")

    # Hypothesis 2
    st.write("## Hypothesis 2 — Luxury vs Economy Brands")
    st.write(
        "H0: There is no significant price difference between "
        "luxury and economy brands.\n\n"
        "H1: Luxury car brands have significantly higher prices "
        "than economy brands."
    )
    avg_price_brand = (
        df.groupby('CarBrand')['price']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_price_brand['brand_type'] = avg_price_brand['price'].apply(
        lambda x: 'Luxury' if x > 20000 else 'Economy'
    )
    fig2 = px.bar(
        avg_price_brand, x='CarBrand', y='price',
        color='brand_type',
        title='Average Price by Car Brand (Luxury vs Economy)',
        labels={'CarBrand': 'Car Brand', 'price': 'Average Price (USD)'},
        color_discrete_map={'Luxury': 'darkblue', 'Economy': 'lightblue'}
    )
    st.plotly_chart(fig2)
    st.caption(
        "Each bar shows the average price for one car brand, sorted "
        "from most to least expensive. Brands are colour-coded: dark "
        "blue means the brand's average price is above $20,000 "
        "(classified here as 'Luxury'), light blue means it's below "
        "that ('Economy'). To check whether this price gap is a real "
        "pattern rather than random chance, we ran a **t-test** — a "
        "statistical test that compares the average of two groups. "
        "The result (p-value = 0.0000, far below the usual 0.05 "
        "cut-off) tells us the difference is highly unlikely to be "
        "due to chance."
    )

    st.success(
        "✅ Answering the question 'Do luxury brands have "
        "significantly higher prices than economy brands?' — "
        "Hypothesis 2 confirmed: Luxury brands are significantly "
        "more expensive (p-value = 0.0000)"
    )

    st.write("---")

    # Hypothesis 3
    st.write("## Hypothesis 3 — Diesel vs Petrol Cars")
    st.write(
        "H0: There is no significant price difference between "
        "diesel and petrol cars.\n\n"
        "H1: Diesel cars have significantly higher prices than "
        "petrol cars."
    )
    st.warning(
        "⚠️ Dataset imbalance: 185 gas cars vs 20 diesel cars — "
        "results should be interpreted with caution."
    )

    avg_price_fuel = df.groupby('fueltype')['price'].mean().reset_index()
    fig3a = px.bar(
        avg_price_fuel, x='fueltype', y='price',
        title='Average Price by Fuel Type (Imbalanced — 185 vs 20)',
        labels={'fueltype': 'Fuel Type', 'price': 'Average Price (USD)'},
        color='fueltype',
        color_discrete_map={'gas': 'lightblue', 'diesel': 'darkblue'}
    )
    st.plotly_chart(fig3a)
    st.caption(
        "This chart compares the average price of diesel vs gas cars "
        "using all the data we have. But there's a problem: we only "
        "have 20 diesel cars compared to 185 gas cars, so this "
        "comparison isn't fair — a handful of unusually priced diesel "
        "cars could easily skew the average in a way that wouldn't "
        "happen with a bigger sample. This is a **sample size** issue "
        "— a core statistical concept describing how the number of "
        "observations in a group affects how much we can trust its "
        "average."
    )

    gas_sample = (
        df[df['fueltype'] == 'gas']['price'].sample(n=20, random_state=42)
    )
    diesel_sample = df[df['fueltype'] == 'diesel']['price']
    balanced_df = pd.DataFrame({
        'price': pd.concat([diesel_sample, gas_sample]),
        'fueltype': ['diesel'] * 20 + ['gas'] * 20
    })
    avg_balanced = balanced_df.groupby('fueltype')['price'].mean()
    avg_balanced = avg_balanced.reset_index()

    fig3b = px.bar(
        avg_balanced, x='fueltype', y='price',
        title='Average Price by Fuel Type (Balanced Sample — 20 vs 20)',
        labels={'fueltype': 'Fuel Type', 'price': 'Average Price (USD)'},
        color='fueltype',
        color_discrete_map={'gas': 'lightblue', 'diesel': 'darkblue'}
    )
    st.plotly_chart(fig3b)
    st.caption(
        "To fix the unfair comparison above, we randomly picked just "
        "20 gas cars — the same number as diesel cars — so both "
        "groups are equally sized. This is called **balanced "
        "sampling**. Comparing like-for-like this way gives a more "
        "trustworthy picture than the first chart."
    )

    st.error(
        "❌ Answering the question 'Do diesel cars have significantly "
        "higher prices than petrol cars?' — Hypothesis 3 rejected: "
        "With balanced sampling, there is no statistically "
        "significant price difference between diesel and gas cars "
        "(p-value = 0.8659, well above the 0.05 cut-off, meaning any "
        "difference we see is likely just due to chance)"
    )

    st.write("---")

    # Hypothesis 4
    st.write("## Hypothesis 4 — Body Style vs Price")
    st.write(
        "H0: Car body style has no significant influence on price.\n\n"
        "H1: Car body style significantly influences car price."
    )
    fig4 = px.box(
        df, x='carbody', y='price',
        title='Price Distribution by Car Body Style',
        labels={'carbody': 'Body Style', 'price': 'Price (USD)'},
        color='carbody'
    )
    st.plotly_chart(fig4)
    st.caption(
        "This is called a **box plot** — each box summarises the "
        "price spread for one body style. The line inside the box is "
        "the **median** (the middle price when all values are sorted "
        "— half the cars cost more, half cost less). The box itself "
        "covers the middle 50% of prices, and the thin lines "
        "extending out ('whiskers') show the typical range beyond "
        "that. Dots sitting far above or below the whiskers are "
        "**outliers** — unusually priced cars for that body style. "
        "Median, quartiles and outliers are core **descriptive "
        "statistics** — they summarise a dataset's spread and central "
        "tendency without assuming a specific distribution shape."
    )

    st.success(
        "✅ Answering the question 'Does car body style "
        "significantly influence price?' — Hypothesis 4 confirmed "
        "(ANOVA test): Body style significantly influences price "
        "(p-value = 0.0000). Convertibles are the most expensive."
    )

    st.write("---")

    # Feature Importance
    st.write("## Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': [
            'enginesize', 'curbweight', 'cylindernumber', 'carwidth',
            'horsepower', 'citympg', 'drivewheel', 'highwaympg'
        ],
        'Importance': [0.20, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.07]
    }).sort_values('Importance', ascending=True)
    fig5 = px.bar(
        importance_df, x='Importance', y='Feature',
        orientation='h',
        title='Feature Importance — ExtraTreesRegressor'
    )
    st.plotly_chart(fig5)
    st.caption(
        "This chart shows how much each car feature contributed to "
        "the machine learning model's price predictions, based on a "
        "score between 0 and 1 (all scores add up to roughly 1.0, or "
        "100%). The longer the bar, the more that feature influenced "
        "the model's decisions. For example, enginesize at 0.20 means "
        "engine size accounted for about 20% of the model's overall "
        "'reasoning' when predicting price — the single biggest "
        "factor. This ranking is called **feature importance** — a "
        "key concept for understanding and explaining what drives a "
        "model's predictions."
    )

    st.write("---")

    # Model Performance
    st.write("## Model Performance")
    st.write("* **Best model:** ExtraTreesRegressor")
    st.write("* **Test R²:** 0.911")
    st.write("* **Test RMSE:** $2,619")
    st.write("* **Test MAE:** $1,572")
    st.caption(
        "**R²** shows how much of the variation in car prices the "
        "model can explain, from 0 (no better than guessing) to 1 "
        "(perfect predictions). A score of 0.911 means the model "
        "explains about 91% of what drives price. **RMSE** and "
        "**MAE** both measure the model's typical prediction error in "
        "dollars — MAE is the average error size (\\$1,572), while "
        "RMSE penalises larger errors more heavily. Both being "
        "reasonably close together (and both far smaller than the "
        "average car price of \\$13,276) indicates the model performs "
        "consistently well, without a few huge mistakes skewing the "
        "results."
    )

    st.write("---")

    # Limitations
    st.write("## Limitations")
    st.warning(
        "* Small dataset (205 rows) — results should be interpreted "
        "with caution\n"
        "* Imbalanced fuel type (185 gas vs 20 diesel) — explains "
        "why H3 was rejected\n"
        "* Mild overfitting (train R²=0.998 vs test R²=0.911)\n"
        "* T-test for H2 was run on brand averages (n=22) rather "
        "than individual cars"
    )

    st.write("---")

    # Business Recommendations
    st.write("## Business Recommendations")
    st.success(
        "* Manufacturers targeting the US luxury segment should "
        "prioritize engine size and body style (convertibles) in "
        "pricing strategy\n"
        "* European brands command a significant price premium\n"
        "* Diesel engines are not a significant price differentiator "
        "in the US market"
    )