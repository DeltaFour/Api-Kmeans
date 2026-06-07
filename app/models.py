from pydantic import BaseModel
from typing import List


class UserCluster(BaseModel):
    userId: str
    cluster: int


class Centroid(BaseModel):
    cluster: int
    latePercentage: float
    averageLateMinutes: float
    maxLateMinutes: float
    totalAbsences: float
    totalWorkedDays: float


class CompanyClassification(BaseModel):
    companyId: str
    userClusters: List[UserCluster]
    centroids: List[Centroid]


class ClassificationPayload(BaseModel):
    companies: List[CompanyClassification]