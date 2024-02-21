from fastapi import FastAPI

app = FastAPI()

@app.get("/user")
def get_user():
    user_data = {
    "member_id": "123456",
    "first_name": "John",
    "last_name": "Doe",
    "policy_number": "P12345",
    "policy_type": "Health Insurance",
    "deductible": 1000,
    "out_of_pocket_max": 5000,
    "copay for pcp": 20,
    "copay for specialist": 40,
    "inpatient_hospital": "80% coverage after deductible",
    "outpatient_surgery": "90% coverage after deductible",
    "prescription_drugs": "Tiered copay based on formulary",
    "preventive_services": "100% coverage (no copay)" ,
    "network_type": "Preferred Provider Organization (PPO)",
    "annual_wellness_visit": "Covered at 100%",
    "telehealth_services": "Available with copay",
    "emergency_room": "80% coverage after deductible"
  }
    return user_data
