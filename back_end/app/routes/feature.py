from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from app.schemas.feature import FeatureRow, FeatureRowRetrieve
from app.client import prisma_client as prisma
from datetime import date, datetime, time
import redis
import asyncio

router = APIRouter()

# Redis connection function
def get_redis_client():
    try:
        redis_client = redis.Redis(host='host.docker.internal', port=6379, db=0)
        # Test connection
        redis_client.ping()
        return redis_client
    except redis.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Redis")

@router.delete("/features/clear")
async def clear_features():
    await prisma.creditcardfeature.delete_many()
    return {"status": "success", "message": "All features deleted."}

@router.post("/features/batch")
async def ingest_features(features: List[FeatureRow]):
    try:
        tasks = [
            prisma.creditcardfeature.create(
                data={
                    "cc_num": f.cc_num,
                    "amt": f.amt,
                    "hour_of_day": f.hour_of_day,
                    "day_of_week": f.day_of_week,
                    "age_at_txn": f.age_at_txn,
                    "distance_to_merchant": f.distance_to_merchant,
                    "txn_count_last_10_min": f.txn_count_last_10_min,
                    "avg_amt_last_1_hour": f.avg_amt_last_1_hour,
                    "category": f.category,
                    "gender": f.gender,
                    "city_pop": f.city_pop,
                    "feature_timestamp": f.trans_date_trans_time,
                    "is_fraud": f.is_fraud
                }
            )
            for f in features
        ]

        await asyncio.gather(*tasks)  # Run them concurrently

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting features: {str(e)}")

    return {"status": "success", "inserted": len(features)}

# sample : http://localhost:8000/api/v1/features/by-ccnum/60422928733
@router.get("/features/by-ccnum/{cc_num}", response_model=List[FeatureRowRetrieve])
async def get_features_by_ccnum(cc_num: str):
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
    
@router.get("/redis/transactions/bulk", response_model=List[Dict[str, Any]])
async def get_bulk_transactions(limit: int = 100_000):
    """
    Fetch up to `limit` transactions from Redis across all cards.
    """
    redis_client = get_redis_client()
    keys = redis_client.scan_iter("txn:*:data:*")

    records = []
    count = 0
    for txn_key in keys:
        txn_data = redis_client.hgetall(txn_key)
        clean_data = {k.decode('utf-8'): try_cast(v) for k, v in txn_data.items()}
        records.append(clean_data)
        count += 1
        if count >= limit:
            break

    return records

@router.get("/redis/transactions/{cc_num}", response_model=Dict[str, Any])
async def get_redis_transactions(cc_num: str):
    """
    Get the latest transaction data and statistics for a credit card from Redis
    """
    redis_client = get_redis_client()
    
    key_prefix = f"txn:{cc_num}"
    stats_key = f"{key_prefix}:stats"
    
    if not redis_client.exists(stats_key):
        raise HTTPException(status_code=404, detail="No transaction data found for this credit card number")
    
    stats = redis_client.hgetall(stats_key)
    
    timeline_key = f"{key_prefix}:timeline"
    recent_transactions = []

    if redis_client.exists(timeline_key):
        # Get the 5 most recent transaction IDs
        recent_txn_ids = redis_client.zrevrange(timeline_key, 0, 4)
        string_data = [b.decode('utf-8') for b in recent_txn_ids]
        
        for txn_id in string_data:
            txn_key = f"{key_prefix}:data:{txn_id}"
            if redis_client.exists(txn_key):
                txn_data = redis_client.hgetall(txn_key)
                recent_transactions.append(txn_data)
    
    return {
        "card_number": cc_num,
        "stats": stats,
        "recent_transactions": recent_transactions
    }


def try_cast(value):
    v = value.decode('utf-8')
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except ValueError:
        return v

@router.post("/features/sync")
async def sync_features_from_redis():
    try:
        redis_client = get_redis_client()

        features = await prisma.creditcardfeature.find_many()

        updated = 0
        skipped = 0

        for row in features:
            row_dict = row.model_dump()
            cc_num = row_dict['cc_num']
            redis_key = f"txn:{cc_num}:stats"

            if redis_client.exists(redis_key):
                static_data = {
                    k: str(v) for k, v in row_dict.items()
                    if k in ['hour_of_day', 'day_of_week', 'age_at_txn', 'category', 'gender', 'city_pop']
                    and v is not None
                }
                redis_client.hset(redis_key, mapping=static_data)
                updated += 1
            else:
                skipped += 1

        return {
            "status": "success",
            "updated": updated,
            "skipped": skipped,
            "message": f"Synced {updated} records. Skipped {skipped} missing Redis keys."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing features: {str(e)}")