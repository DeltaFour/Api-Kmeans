import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader

from services.kmeans_service import execute_kmeans

API_KEY_HEADER_NAME = "X-API-Key"

# Chave que o chamador (cron) precisa apresentar para disparar o K-Means.
# É distinta da API_KEY usada pelo serviço para chamar o backend .NET.
CRON_API_KEY = os.getenv("CRON_API_KEY")

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

app = FastAPI(
    title="Punctuality KMeans Service"
)


def require_api_key(provided_key: str = Depends(api_key_header)):

    if not CRON_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CRON_API_KEY não configurada no serviço."
        )

    if not provided_key or not secrets.compare_digest(
        provided_key,
        CRON_API_KEY
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave de API inválida ou ausente."
        )


@app.post(
    "/executar-kmeans",
    dependencies=[Depends(require_api_key)]
)
def executar_kmeans():

    try:
        return execute_kmeans()

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
