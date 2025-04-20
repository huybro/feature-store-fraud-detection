from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.schemas.feature import FeatureRow
from app.client import prisma_client as prisma

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