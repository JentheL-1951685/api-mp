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
    "http://localhost:5173",
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

# Global variable to store the trained Isolation Forest models
isolation_forest_models = {
    "monthly": {},  # Models for monthly_cost and monthly_kWh
    "charge": {}    # Models for price and charge_added
}

@app.on_event("startup")
async def train_isolation_forest_models():
    global isolation_forest_models
    print("Updating anomaly models...")

    # Train models for "monthly_cost" and "monthly_kWh"
    monthly_query = text("""
        SELECT
            u.id AS user_id,
            f.id AS fleet_id,
            DATE_FORMAT(c.finished_charging_at, '%Y-%m') AS month,
            SUM(c.price) AS monthly_cost,
            SUM(c.charge_added) AS monthly_kWh
        FROM
            users AS u
            JOIN cars AS ca ON u.id = ca.user_id
            JOIN brands AS b ON ca.brand_id = b.id
            JOIN car_fleet AS cf ON ca.id = cf.car_id
            JOIN fleets AS f ON cf.fleet_id = f.id
            JOIN charges AS c ON ca.id = c.car_id
        WHERE
            u.currency_code = 'EUR' 
            AND b.name != 'skoda'
        GROUP BY
            u.id, f.id, DATE_FORMAT(c.finished_charging_at, '%Y-%m')
    """)

    df = pd.read_sql(monthly_query, engine)

    if not df.empty:
        df['price_per_kWh'] = df['monthly_cost'] / df['monthly_kWh']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        anomaly_inputs = ['monthly_cost', 'monthly_kWh']
        df[anomaly_inputs] = df[anomaly_inputs].clip(upper=1e10)
        df.dropna(subset=anomaly_inputs, inplace=True)

        for fleet_id in df['fleet_id'].unique():
            fleet_df = df[df['fleet_id'] == fleet_id].copy()
            isolation_forest_models["monthly"][fleet_id] = {}

            for user_id in fleet_df['user_id'].unique():
                person_df = fleet_df[fleet_df['user_id'] == user_id].copy()
                model_IF = IsolationForest(contamination='auto', random_state=10)
                model_IF.fit(person_df[anomaly_inputs])
                isolation_forest_models["monthly"][fleet_id][user_id] = model_IF

    # Train models for "price" and "charge_added"
    charge_query = text("""
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
            JOIN brands AS b ON ca.brand_id = b.id
            JOIN car_fleet AS cf ON ca.id = cf.car_id
            JOIN fleets AS f ON cf.fleet_id = f.id
            JOIN charges AS c ON ca.id = c.car_id
        WHERE
            u.currency_code = 'EUR' 
            AND b.name != 'skoda'
    """)

    df = pd.read_sql(charge_query, engine)

    if not df.empty:
        anomaly_inputs = ['price', 'charge_added']
        df.dropna(subset=anomaly_inputs, inplace=True)

        for fleet_id in df['fleet_id'].unique():
            fleet_df = df[df['fleet_id'] == fleet_id].copy()
            isolation_forest_models["charge"][fleet_id] = {}

            for user_id in fleet_df['user_id'].unique():
                person_df = fleet_df[fleet_df['user_id'] == user_id].copy()
                model_IF = IsolationForest(contamination='auto', random_state=10)
                model_IF.fit(person_df[anomaly_inputs])
                isolation_forest_models["charge"][fleet_id][user_id] = model_IF

    print("Anomaly models updated.")
@app.get("/api/v1/fleets")
async def get_fleets():
    query = """
        SELECT
            id, country, city
        FROM
            fleets
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient='records')


@app.get("/api/v1/fleets/{fleet_id}/anomaly/{month}")
async def get_fleet_anomaly_for_month(fleet_id: int, month: str = Path(..., title="The month for which to retrieve anomaly data (YYYY-MM)")):
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
            JOIN brands AS b ON ca.brand_id = b.id
            JOIN car_fleet AS cf ON ca.id = cf.car_id
            JOIN fleets AS f ON cf.fleet_id = f.id
            JOIN charges AS c ON ca.id = c.car_id
        WHERE
            u.currency_code = 'EUR'
            AND f.id = :fleet_id
            AND b.name != 'skoda'
        GROUP BY
            u.id, f.id, DATE_FORMAT(c.finished_charging_at, '%Y-%m')
    """)

    df = pd.read_sql(query, engine, params={"fleet_id": fleet_id})

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified fleet.")

    df['price_per_kWh'] = df['monthly_cost'] / df['monthly_kWh']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    anomaly_inputs = ['monthly_cost', 'monthly_kWh']
    df[anomaly_inputs] = df[anomaly_inputs].clip(upper=1e10)
    df.dropna(subset=anomaly_inputs, inplace=True)

    person_dfs = []
    for user_id in df['user_id'].unique():
        if fleet_id in isolation_forest_models["monthly"] and user_id in isolation_forest_models["monthly"][fleet_id]:
            person_df = df[df['user_id'] == user_id].copy()
            model_IF = isolation_forest_models["monthly"][fleet_id][user_id]
            person_df['anomaly_scores'] = model_IF.decision_function(person_df[anomaly_inputs])
            person_df['anomaly'] = model_IF.predict(person_df[anomaly_inputs])
            person_dfs.append(person_df)

    result_df = pd.concat(person_dfs, ignore_index=True)
    result_df = result_df[result_df['month'] == month]

    if result_df.empty:
        raise HTTPException(status_code=404, detail=f"No anomaly data found for month {month}.")

    result_df = result_df[['user_id', 'fleet_id', 'monthly_cost', 'monthly_kWh', 'price_per_kWh', 'anomaly_scores', 'anomaly']]
    return result_df.to_dict(orient='records')


@app.get("/api/v1/fleets/{fleet_id}/anomaly/{month}/{user_id}")
async def get_fleet_anomaly_for_user(fleet_id: int, user_id: int, month: str = Path(..., title="The month for which to retrieve anomaly data (YYYY-MM)")):
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
            JOIN brands AS b ON ca.brand_id = b.id
            JOIN car_fleet AS cf ON ca.id = cf.car_id
            JOIN fleets AS f ON cf.fleet_id = f.id
            JOIN charges AS c ON ca.id = c.car_id
        WHERE
            u.currency_code = 'EUR'
            AND f.id = :fleet_id
            AND u.id = :user_id
            AND DATE_FORMAT(c.finished_charging_at, '%Y-%m') = :month
            AND b.name != 'skoda'
    """)

    df = pd.read_sql(query, engine, params={"fleet_id": fleet_id, "user_id": user_id, "month": month})

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    df['month'] = pd.to_datetime(df['finished_charging_at']).dt.to_period('M')
    df['charging_time'] = df['finished_charging_at'] - df['started_charging_at']

    anomaly_inputs = ['price', 'charge_added']
    if fleet_id in isolation_forest_models["charge"] and user_id in isolation_forest_models["charge"][fleet_id]:
        model_IF = isolation_forest_models["charge"][fleet_id][user_id]
        df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
        df['anomaly'] = model_IF.predict(df[anomaly_inputs])

    result_df = df[['month', 'user_id', 'price', 'charge_added', 'started_charging_at', 'finished_charging_at', 'charging_time', 'anomaly_scores', 'anomaly']]
    return result_df.to_dict(orient='records')