import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor():
    num_features = ["reading_score", "writing_score"]
    cat_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )
    return preprocessor


if __name__ == "__main__":
    # ---- Load dataset ----
    df = pd.read_csv("notebook\data\stud.csv")   # 👈 update with your dataset path

    # ---- Split features/target ----
    X = df.drop("math_score", axis=1)
    y = df["math_score"]

    # ---- Preprocess ----
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # ---- Save preprocessor ----
    save_object("artifacts/preprocessor.pkl", preprocessor)

    # ---- Train model ----
    model = RandomForestRegressor()
    model.fit(X_transformed, y)

    # ---- Save model ----
    save_object("artifacts/model.pkl", model)

    print("✅ Training complete. Preprocessor and model saved in 'artifacts/' folder.")
