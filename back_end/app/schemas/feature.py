from pydantic import BaseModel

class FeatureRow(BaseModel):
    cc_num: int
    amt: float
    hour_of_day: int
    day_of_week: int
    age_at_txn: float
    distance_to_merchant: float
    txn_count_last_10_min: int
    avg_amt_last_1_hour: float
    category: str
    gender: str
    city_pop: int
    is_fraud: int