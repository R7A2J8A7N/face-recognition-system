# from fastapi import APIRouter
# from src.api.dependencies import get_engine

# router = APIRouter(prefix="/enroll", tags=["Enrollment"])


# @router.post("/")
# def enroll(dataset_path: str):

#     engine = get_engine()

#     report = engine.enroll_dataset(dataset_path)

#     return report
