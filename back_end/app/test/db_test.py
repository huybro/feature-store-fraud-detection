import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def prisma_mock():
    with patch('app.main.prisma', new_callable=AsyncMock) as mock:
        mock.response = AsyncMock()
        mock.responsecomponent = AsyncMock()
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.is_connected = MagicMock(return_value=True)
        yield mock


#Test root endpoint
@pytest.mark.asyncio
async def test_root(client):
    response = client.get("/")
    assert response.json() == {"message": "Your app is working!"}