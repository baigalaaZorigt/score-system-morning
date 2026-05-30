import os
import sys
import numpy as np
import joblib
import sklearn.ensemble
from flask import Flask, render_template, request

# Compatibility shim: the pickle was saved with sklearn's internal module path
# that may differ across versions.
sys.modules.setdefault("sklearn.ensemble._gb", sklearn.ensemble)

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_scoring_model.pkl")

data = joblib.load(MODEL_PATH)
# The pickle is a dict {"model": <estimator>, ...}; unwrap if needed.
model = data["model"] if isinstance(data, dict) and "model" in data else data


def get_decision(score: float) -> tuple[str, str]:
    if score >= 700:
        return "Зөвшөөрөх", "approved"
    elif score >= 450:
        return "Гар шалгалт", "review"
    else:
        return "Татгалзах", "rejected"


def build_features(monthly_income: float, employment_type_encoded: int,
                   employment_years: float, requested_amount: float) -> np.ndarray:
    """
    Build the 8 features the model was trained on:
      monthly_income, employment_years, requested_amount,
      amount_to_income_ratio, annual_dti, log_income,
      log_amount, employment_type_encoded
    """
    amount_to_income_ratio = requested_amount / monthly_income if monthly_income > 0 else 0.0
    annual_dti = requested_amount / (monthly_income * 12) if monthly_income > 0 else 0.0
    log_income = float(np.log1p(monthly_income))
    log_amount = float(np.log1p(requested_amount))

    return np.array([[
        monthly_income,
        employment_years,
        requested_amount,
        amount_to_income_ratio,
        annual_dti,
        log_income,
        log_amount,
        employment_type_encoded,
    ]])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        monthly_salary = float(request.form["monthly_salary"])
        employment_status = int(request.form["employment_status"])
        years_worked = float(request.form["years_worked"])
        loan_amount = float(request.form["loan_amount"])

        if monthly_salary <= 0 or years_worked < 0 or loan_amount <= 0:
            raise ValueError("Утгууд хүчингүй байна.")

        features = build_features(monthly_salary, employment_status, years_worked, loan_amount)

        # Use predict_proba if available (classifier) to derive a 0-1000 score.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            # Take probability of the positive / highest class as the raw score.
            raw_score = float(proba[-1]) * 1000
        else:
            raw_score = float(model.predict(features)[0])

        score = int(round(np.clip(raw_score, 0, 1000)))

        decision, status = get_decision(score)

        return render_template(
            "index.html",
            score=score,
            decision=decision,
            status=status,
            monthly_salary=monthly_salary,
            employment_status=employment_status,
            years_worked=years_worked,
            loan_amount=loan_amount,
        )

    except (ValueError, KeyError) as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", debug=True, port=port)
