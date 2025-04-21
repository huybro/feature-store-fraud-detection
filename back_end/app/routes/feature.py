from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
from app.schemas.feature import FeatureRow, FeatureRowRetrieve
from app.client import prisma_client as prisma
from datetime import date, datetime, time

router = APIRouter()

@router.post("/features/batch")
async def ingest_features(features: List[FeatureRow]):
    if not prisma.is_connected():
        await prisma.connect()

    try:
        for feature in features:
            await prisma.creditcardfeature.create(
                data={
                    "cc_num": feature.cc_num,
                    "amt": feature.amt,
                    "hour_of_day": feature.hour_of_day,
                    "day_of_week": feature.day_of_week,
                    "age_at_txn": feature.age_at_txn,
                    "distance_to_merchant": feature.distance_to_merchant,
                    "txn_count_last_10_min": feature.txn_count_last_10_min,
                    "avg_amt_last_1_hour": feature.avg_amt_last_1_hour,
                    "category": feature.category,
                    "gender": feature.gender,
                    "city_pop": feature.city_pop,
                    "is_fraud": feature.is_fraud
                    # feature_timestamp will default to now() from Prisma schema
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting features: {str(e)}")

    return {"status": "success", "inserted": len(features)}

# sample : http://localhost:8000/api/v1/features/by-ccnum/60422928733
@router.get("/features/by-ccnum/{cc_num}", response_model=List[FeatureRowRetrieve])
async def get_features_by_ccnum(cc_num: str):
    if not prisma.is_connected():
        await prisma.connect()

    try:
        features = await prisma.creditcardfeature.find_many(
            where={"cc_num": int(cc_num)}
        )
        if not features:
            raise HTTPException(status_code=404, detail="No features found for this cc_num.")
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {str(e)}")

# sample : http://localhost:8000/api/v1/features/by-date?start=2025-04-21&end=2025-04-25
@router.get("/features/by-date", response_model=List[FeatureRowRetrieve])
async def get_features_by_date_range(
    start: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: date = Query(..., description="End date (YYYY-MM-DD)")
):
    if not prisma.is_connected():
        await prisma.connect()

    if start > end:
        raise HTTPException(status_code=400, detail="Start date must be on or before end date.")

    # Convert to full-day datetime range
    start_datetime = datetime.combine(start, time.min)  # 00:00:00
    end_datetime = datetime.combine(end, time.max)      # 23:59:59.999999

    try:
        features = await prisma.creditcardfeature.find_many(
            where={
                "feature_timestamp": {
                    "gte": start_datetime,
                    "lte": end_datetime
                }
            }
        )
        if not features:
            raise HTTPException(status_code=404, detail="No features found in the given date range.")
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {str(e)}")