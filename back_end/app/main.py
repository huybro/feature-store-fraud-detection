from fastapi import Depends, FastAPI, HTTPException
from prisma import Prisma
from typing import List

from app.routes.feature import router as feature_router
from app.dependencies import use_logging
from app.middleware import LoggingMiddleware
from app.client import prisma_client as prisma, connect_db, disconnect_db

app = FastAPI(prefix="/api/v1")
app.add_middleware(LoggingMiddleware, fastapi=app)

# prisma = Prisma(auto_register=True)

@app.get("/")
async def root(logger=Depends(use_logging)):
    logger.info("Handling your request")
    return {"message": "Your app is working!"}


@app.on_event("startup")
async def startup() -> None:
    await prisma.connect()

@app.on_event("shutdown")
async def shutdown() -> None:
    if prisma.is_connected():
        await prisma.disconnect()


app.include_router(feature_router, prefix="/api/v1")

