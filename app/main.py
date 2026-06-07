from fastapi import FastAPI, HTTPException

from services.kmeans_service import execute_kmeans

app = FastAPI(
    title="Punctuality KMeans Service"
)

@app.post("/executar-kmeans")
def executar_kmeans():

    try:
        return execute_kmeans()

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )