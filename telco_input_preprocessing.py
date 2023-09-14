import numpy as np
import pandas as pd

# load the data preprocessors from disk
import joblib

dv = joblib.load(r"./notebook/dictvectorizer.sav")
scaler = joblib.load(r"./notebook/scaler.sav")


def preprocess(df, option):
    def binary_map(columns):
        return columns.map({"Yes": 1, "No": 0})

    # Drop values based on operational options
    if option == "Online":

        df["age"] = df["age"].apply(
            lambda record: "61_and_above"
            if record >= 61
            else "31_-_60"
            if record >= 31 and record < 61
            else "30_and_below"
        )

        # Encode binary categorical features
        binary_list = [
            "married",
            "dependents",
            "phone_service",
            "online_security",
            "online_backup",
            "device_protection_plan",
            "premium_tech_support",
            "streaming_tv",
            "streaming_movies",
            "streaming_music",
            "unlimited_data",
            "paperless_billing",
        ]

        df[binary_list] = df[binary_list].apply(binary_map)
        # Preparing the test set
        numerical = [
            "number_of_referrals",
            "tenure_in_months",
            "avg_monthly_long_distance_charges",
            "avg_monthly_gb_download",
            "monthly_charge",
        ]

        related_cats = [
            "satisfaction_score",
            "dependents",
            "contract",
            "internet_service",
            "payment_method",
            "paperless_billing",
            "online_security",
            "unlimited_data",
            "premium_tech_support",
            "married",
            "age",
            "online_backup",
            "device_protection_plan",
            "streaming_tv",
            "streaming_movies",
            "streaming_music",
        ]
        nums_test = df[numerical].values
        nums_test_scaled = scaler.transform(nums_test)

        data_values_test = np.column_stack([nums_test_scaled, df[related_cats].values])
        column_names = numerical + related_cats
        df_test_scaled = pd.DataFrame(data_values_test, columns=column_names)

        # Creating the test dictionary of records
        dicts_test = df_test_scaled.to_dict(orient="records")
        # Transforming the test dictionary using the trained DictVectorizer by the full_train
        X_test = dv.transform(dicts_test)

    elif option == "Batch":
        # Making the column names consistent
        df.columns = df.columns.str.replace(" ", "_").str.lower()

        binary_list = [
            "married",
            "dependents",
            "phone_service",
            "online_security",
            "online_backup",
            "device_protection_plan",
            "premium_tech_support",
            "streaming_tv",
            "streaming_movies",
            "streaming_music",
            "unlimited_data",
            "paperless_billing",
        ]
        df[binary_list].fillna('Yes', inplace=True)

        df[binary_list] = df[binary_list].apply(binary_map)

        df["age"] = df["age"].apply(
            lambda record: "61_and_above"
            if record >= 61
            else "31_-_60"
            if record >= 31 and record < 61
            else "30_and_below"
        )

        # Deleting the customer_id column
        del df["customer_id"]

        # We also clean the values of each categorical column
        # making them lowercase and fixing the spaces
        strings = ["gender", "internet_service", "contract", "payment_method"]

        for col in strings:
            df[col] = df[col].str.lower().str.replace(" ", "_")

        df = df[
            [
                "dependents",
                "number_of_referrals",
                "tenure_in_months",
                "avg_monthly_long_distance_charges",
                "avg_monthly_gb_download",
                "monthly_charge",
                "satisfaction_score",
                "contract",
                "internet_service",
                "payment_method",
                "paperless_billing",
                "online_security",
                "unlimited_data",
                "premium_tech_support",
                "married",
                "age",
                "online_backup",
                "device_protection_plan",
                "streaming_tv",
                "streaming_movies",
                "streaming_music",
            ]
        ]

        numerical = [
            "number_of_referrals",
            "tenure_in_months",
            "avg_monthly_long_distance_charges",
            "avg_monthly_gb_download",
            "monthly_charge",
        ]

        related_cats = [
            "satisfaction_score",
            "dependents",
            "contract",
            "internet_service",
            "payment_method",
            "paperless_billing",
            "online_security",
            "unlimited_data",
            "premium_tech_support",
            "married",
            "age",
            "online_backup",
            "device_protection_plan",
            "streaming_tv",
            "streaming_movies",
            "streaming_music",
        ]
        nums_test = df[numerical].values
        nums_test_scaled = scaler.transform(nums_test)

        data_values_test = np.column_stack([nums_test_scaled, df[related_cats].values])
        column_names = numerical + related_cats
        df_test_scaled = pd.DataFrame(data_values_test, columns=column_names)

        # Creating the test dictionary of records
        dicts_test = df_test_scaled.to_dict(orient="records")
        # Transforming the test dictionary using the trained DictVectorizer by the full_train
        X_test = dv.transform(dicts_test)

    else:
        print("Incorrect operational options")

    return X_test
