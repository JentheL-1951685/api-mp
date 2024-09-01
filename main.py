from fastapi import FastAPI, HTTPException, Path
import pandas as pd
import sqlalchemy
from sqlalchemy import text
from sklearn.ensemble import IsolationForest
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Define the origins that should be allowed to make CORS requests
origins = [
    "https://main--masterproef-jenthe-lowist.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from the .env file
load_dotenv()

# Retrieve the database URL from environment variables
database_url = os.getenv('DATABASE_URL')

# Create the engine using the environment variable
engine = sqlalchemy.create_engine(database_url)

@app.get("/api/v1/fleets")
async def get_fleets():
    # Query data from the database using the engine directly
    query = """
        SELECT
            id, country, city
        FROM
            fleets
    """
    df = pd.read_sql(query, engine)

    # Convert DataFrame to JSON and return
    return df.to_dict(orient='records')

@app.get("/api/v1/fleets/{fleet_id}/anomaly/{month}")
async def get_fleet_anomaly_for_month(fleet_id: int, month: str = Path(..., title="The month for which to retrieve anomaly data (YYYY-MM)")):
    # Query data from the database using the engine directly
    query = text("""
        SELECT
            u.id AS user_id,
            f.id AS fleet_id,
            DATE_FORMAT(c.finished_charging_at, '%Y-%m') AS month,
            SUM(c.price) AS monthly_cost,
            SUM(c.charge_added) AS monthly_kWh
        FROM
            users AS u
            JOIN cars AS ca ON u.id = ca.user_id
            JOIN car_fleet AS cf ON ca.id = cf.car_id
            JOIN fleets AS f ON cf.fleet_id = f.id
            JOIN charges AS c ON ca.id = c.car_id
        WHERE
            u.currency_code = 'EUR'
            AND f.id = :fleet_id
        GROUP BY
            u.id, f.id, DATE_FORMAT(c.finished_charging_at, '%Y-%m')
    """)

    df = pd.read_sql(query, engine, params={"fleet_id": fleet_id})

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified fleet.")

    # Calculate the price_per_kWh for each month
    df['price_per_kWh'] = df['monthly_cost'] / df['monthly_kWh']

    # Building an Isolation Forest Model
    anomaly_inputs = ['monthly_cost', 'monthly_kWh']

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle extremely large values by clipping or replacing them with a specific value
    max_values = df[anomaly_inputs].max()
    threshold = 1e10
    df[anomaly_inputs] = df[anomaly_inputs].clip(upper=threshold)
    df.dropna(subset=anomaly_inputs, inplace=True)

    # Initialize an empty list to store all person_df
    person_dfs = []

    # Iterate through each unique user ID
    for user_id in df['user_id'].unique():
        # Create a new DataFrame for the current user with data across multiple months
        person_df = df[df['user_id'] == user_id].copy()

        # Fit the Isolation Forest model on the user's monthly costs across different months
        model_IF = IsolationForest(contamination='auto', random_state=10)
        model_IF.fit(person_df[anomaly_inputs])

        # Predict anomalies for the specific month of interest
        person_df['anomaly_scores'] = model_IF.decision_function(person_df[anomaly_inputs])
        person_df['anomaly'] = model_IF.predict(person_df[anomaly_inputs])

        # Append person_df to person_dfs list
        person_dfs.append(person_df)

    # Concatenate all person_dfs into result_df
    result_df = pd.concat(person_dfs, ignore_index=True)

    # Filter the results by the specified month
    result_df = result_df[result_df['month'] == month]

    # If the result_df is empty after filtering by month, return a 404
    if result_df.empty:
        raise HTTPException(status_code=404, detail=f"No anomaly data found for month {month}.")

    # Select relevant columns
    result_df = result_df[['user_id', 'fleet_id', 'monthly_cost', 'monthly_kWh', 'price_per_kWh', 'anomaly_scores', 'anomaly']]

    # Convert DataFrame to JSON and return
    return result_df.to_dict(orient='records')

@app.get("/api/v1/fleets/{fleet_id}/anomaly/{month}/{user_id}")
async def get_fleet_anomaly_for_user(fleet_id: int, user_id: int, month: str = Path(..., title="The month for which to retrieve anomaly data (YYYY-MM)")):
    # Query data from the database using the engine directly
    query = text("""
        SELECT
            u.id AS user_id,
            f.id AS fleet_id,
            c.price,
            c.charge_added,
            c.finished_charging_at,
            c.started_charging_at
        FROM
            users AS u
            JOIN cars AS ca ON u.id = ca.user_id
            JOIN car_fleet AS cf ON ca.id = cf.car_id
            JOIN fleets AS f ON cf.fleet_id = f.id
            JOIN charges AS c ON ca.id = c.car_id
        WHERE
            u.currency_code = 'EUR'
            AND f.id = :fleet_id
            AND u.id = :user_id
            AND DATE_FORMAT(c.finished_charging_at, '%Y-%m') = :month
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"fleet_id": fleet_id, "user_id": user_id, "month": month})

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    # Extracting month and year from finished_charging_at
    df['month'] = pd.to_datetime(df['finished_charging_at']).dt.to_period('M')

    df['charging_time'] = df['finished_charging_at'] - df['started_charging_at']

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified month")

    # Building an Isolation Forest Model
    anomaly_inputs = ['price', 'charge_added']
    df.dropna()

    # Fit the Isolation Forest model
    model_IF = IsolationForest(contamination='auto', random_state=10)
    model_IF.fit(df[anomaly_inputs])

    # Predict anomalies
    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])

    result_df = df[['month', 'user_id', 'price', 'charge_added', 'started_charging_at', 'finished_charging_at', 'charging_time', 'anomaly_scores', 'anomaly']]

    # Convert DataFrame to JSON and return
    return result_df.to_dict(orient='records')