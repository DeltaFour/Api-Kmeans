import os
import pandas as pd
import requests

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8080"
)

NUMBER_OF_CLUSTERS = int(
    os.getenv(
        "NUMBER_OF_CLUSTERS",
        "3"
    )
)

API_KEY = os.getenv(
    "API_KEY",
    "KMEANS_API_KEY"
)

EXPORT_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/punctuality-metrics/export"
)

CLASSIFICATION_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/punctuality-metrics/classification"
)

FEATURES = [
    "latePercentage",
    "averageLateMinutes",
    "maxLateMinutes",
    "totalAbsences",
    "totalWorkedDays"
]


def _auth_headers():

    if not API_KEY:
        raise Exception(
            "API_KEY não configurada. Defina a variável de ambiente API_KEY."
        )

    return {
        "X-API-Key": API_KEY
    }


def execute_kmeans():

    dataset = _load_dataset()

    if len(dataset) == 0:
        raise Exception(
            "Nenhum registro encontrado para classificação."
        )

    df = pd.DataFrame(dataset)

    if "companyId" not in df.columns:
        raise Exception(
            "O dataset não contém o campo companyId."
        )

    companies_payload = []
    summaries = []

    for company_id, company_df in df.groupby("companyId"):

        result = _process_company(company_id, company_df)

        if result is None:
            summaries.append({
                "companyId": company_id,
                "skipped": True,
                "reason": "registros insuficientes para o número de clusters",
                "records": len(company_df)
            })
            continue

        company_payload, summary = result
        companies_payload.append(company_payload)
        summaries.append(summary)

    if len(companies_payload) == 0:
        raise Exception(
            "Nenhuma empresa possui registros suficientes para classificação."
        )

    _send_classification({"companies": companies_payload})

    return {
        "success": True,
        "totalRecords": len(df),
        "companiesProcessed": len(companies_payload),
        "clusters": NUMBER_OF_CLUSTERS,
        "companies": summaries
    }


def _process_company(company_id, company_df):

    # Sem registros suficientes para formar os clusters pedidos.
    if len(company_df) < NUMBER_OF_CLUSTERS:
        return None

    company_df = company_df.reset_index(drop=True)

    X = company_df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)
    company_df["cluster"] = clusters

    # silhouette só é válido com mais de 1 cluster efetivo.
    silhouette = None
    if len(set(clusters)) > 1:
        silhouette = round(
            float(silhouette_score(X_scaled, clusters)),
            4
        )

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    cluster_mapping = _normalize_cluster_order(company_df)

    company_df["cluster"] = company_df["cluster"].map(cluster_mapping)

    centroids_payload = _build_centroids_payload(
        centroids,
        cluster_mapping
    )

    user_clusters = _build_user_clusters(company_df)

    company_payload = {
        "companyId": company_id,
        "userClusters": user_clusters,
        "centroids": centroids_payload
    }

    summary = {
        "companyId": company_id,
        "skipped": False,
        "records": len(company_df),
        "inertia": round(float(kmeans.inertia_), 4),
        "silhouetteScore": silhouette,
        "userClusters": len(user_clusters),
        "centroids": len(centroids_payload)
    }

    return company_payload, summary


def _load_dataset():

    response = requests.get(
        EXPORT_ENDPOINT,
        headers=_auth_headers(),
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def _send_classification(payload):

    response = requests.put(
        CLASSIFICATION_ENDPOINT,
        json=payload,
        headers=_auth_headers(),
        timeout=60
    )

    response.raise_for_status()


def _build_user_clusters(df):

    result = []

    for row in df.itertuples():

        result.append({
            "userId": row.userId,
            "cluster": int(row.cluster)
        })

    return result


def _build_centroids_payload(
    centroids,
    cluster_mapping
):

    result = []

    for old_cluster, centroid in enumerate(centroids):

        result.append({
            "cluster": int(
                cluster_mapping[old_cluster]
            ),
            "latePercentage": round(
                float(centroid[0]),
                2
            ),
            "averageLateMinutes": round(
                float(centroid[1]),
                2
            ),
            "maxLateMinutes": round(
                float(centroid[2]),
                2
            ),
            "totalAbsences": round(
                float(centroid[3]),
                2
            ),
            "totalWorkedDays": round(
                float(centroid[4]),
                2
            )
        })

    return sorted(
        result,
        key=lambda x: x["cluster"]
    )


def _normalize_cluster_order(df):

    cluster_stats = (
        df.groupby("cluster")
          .agg({
              "latePercentage": "mean"
          })
          .reset_index()
          .sort_values(
              by="latePercentage"
          )
    )

    mapping = {}

    for new_cluster, row in enumerate(
        cluster_stats.itertuples()
    ):
        mapping[row.cluster] = new_cluster

    return mapping
