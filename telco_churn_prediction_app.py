# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# load the model from disk
import joblib

model = joblib.load(r"./notebook/model.sav")
model_threshold = joblib.load(r"./notebook/threshold.sav")

# Import python scripts
from telco_input_preprocessing import preprocess


def main():
    # Setting Application title
    st.title("Telco Customer Churn Prediction App")

    # Setting Application description
    st.markdown(
        """
     :dart:  This Streamlit app is made to predict customer churn in a fictional telecommunication use case.
    The application is functional for both online prediction and batch data prediction. \n
    """
    )
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.snow()

    # Setting Application sidebar default
    image = Image.open("telco-customer-churn.jpg")
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
    )
    st.sidebar.info("This app is created to predict Customer Churn.", icon="ℹ️")
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info(
            "Input data below. Please fill in all values to receive an accurate prediction."
        )
        # Here are the selected features from our dataset
        st.subheader("Demographic data")
        customer_id = st.text_input(label="Customer ID", placeholder="0000-ABCDE")
        gender = st.selectbox("Gender", ("Male", "Female"))
        age = st.number_input("Current Age", min_value=15, max_value=120)
        married = st.selectbox("Is the customer married", ("Yes", "No"))
        dependents = st.selectbox("Does the customer has dependents?", ("Yes", "No"))
        number_of_referrals = st.number_input(
            "Number of Referrals Made", min_value=0, max_value=500
        )

        st.subheader("Payment data")
        tenure_in_months = st.slider(
            "Number of months the customer has stayed with the company",
            min_value=1,
            max_value=200,
        )
        contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
        paperless_billing = st.selectbox("Paperless Billing:", ("Yes", "No"))
        payment_method = st.selectbox(
            "PaymentMethod", ("Mailed check", "Bank Withdrawal", "Credit card")
        )
        monthly_charge = st.number_input(
            "The amount charged to the customer monthly",
            min_value=1.0,
            max_value=1000.0,
            step=1.0,
            format="%.2f",
        )

        st.subheader("Services signed up for")

        phone_service = st.selectbox(
            "Does the customer have phone service?", ("Yes", "No")
        )
        internet_service = st.selectbox(
            "Type of internet service plan (if applicable)",
            ("DSL", "Cable", "Fiber optic", "Not Applicable"),
        )

        avg_monthly_long_distance_charges = st.number_input(
            "Average monthly long distance charges",
            min_value=0.0,
            max_value=1000.0,
            step=1.0,
            format="%.2f",
        )

        avg_monthly_gb_download = st.number_input(
            "Average monthly gig download",
            min_value=0.0,
            max_value=1000.0,
            step=1.0,
            format="%.2f",
        )

        online_security = st.selectbox(
            "Does the customer have online security?", ("Yes", "No")
        )
        online_backup = st.selectbox(
            "Does the customer have online backup?", ("Yes", "No")
        )
        premium_tech_support = st.selectbox(
            "Does the customer have technology support?", ("Yes", "No")
        )
        device_protection_plan = st.selectbox(
            "Does the customer opted for a device protection plan?", ("Yes", "No")
        )
        unlimited_data = st.selectbox(
            "Does the customer pays additional charge for unlimited data?",
            ("Yes", "No"),
        )
        streaming_tv = st.selectbox("Does the customer stream TV?", ("Yes", "No"))
        streaming_movies = st.selectbox(
            "Does the customer stream movies?", ("Yes", "No")
        )
        streaming_music = st.selectbox("Does the customer stream music?", ("Yes", "No"))

        st.subheader("Customer's overall satisfaction score")
        satisfaction_score = st.slider(
            "Rating given by the customer", min_value=1, max_value=5
        )

        data = {
            "customer_id": customer_id,
            "gender": gender,
            "age": age,
            "married": married,
            "dependents": dependents,
            "number_of_referrals": number_of_referrals,
            "tenure_in_months": tenure_in_months,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charge": monthly_charge,
            "phone_service": phone_service,
            "internet_service": internet_service,
            "avg_monthly_long_distance_charges": avg_monthly_long_distance_charges,
            "avg_monthly_gb_download": avg_monthly_gb_download,
            "online_security": online_security,
            "online_backup": online_backup,
            "premium_tech_support": premium_tech_support,
            "device_protection_plan": device_protection_plan,
            "unlimited_data": unlimited_data,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "streaming_music": streaming_music,
            "satisfaction_score": satisfaction_score,
        }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write("Overview of input is shown below")
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # Preprocess inputs
        preprocess_df = preprocess(features_df, "Online")

        online_soft_prediction = model.predict_proba(preprocess_df)[:, 1][0]

        if st.button("Predict"):
            if online_soft_prediction >= model_threshold:
                online_churn_value = 1
                st.warning(
                    f"The customer is leaving. Communicate as soon as possible.",
                    icon="🚨",
                )
            else:
                online_churn_value = 0
                st.success("The customer is staying! Maintain good service.", icon="✅")

            # Download results
            results = pd.DataFrame.from_dict([data])
            results["churn_probability"] = online_soft_prediction
            results["threshold"] = model_threshold
            results["churn_value"] = online_churn_value
            st.download_button(
                label="Download prediction",
                data=results.to_csv(index=None),
                mime="text/csv",
            )

    else:
        st.subheader("Dataset upload")
    
        with open(r"./template/customer_template.zip", "rb") as fp:
            btn = st.download_button(
                label="Get template",
                data=fp,
                file_name="customer_template.zip",
                mime="application/zip",
            )
        uploaded_file = st.file_uploader(
            "Choose a file",
            help="Upload customer data. Save the file in csv format before uploading.",
        )

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            original_data = data.copy()
            original_data.columns = original_data.columns.str.replace(
                " ", "_"
            ).str.lower()
            original_data = original_data[
                [
                    "customer_id",
                    "gender",
                    "age",
                    "married",
                    "dependents",
                    "number_of_referrals",
                    "tenure_in_months",
                    "contract",
                    "paperless_billing",
                    "payment_method",
                    "monthly_charge",
                    "phone_service",
                    "internet_service",
                    "avg_monthly_long_distance_charges",
                    "avg_monthly_gb_download",
                    "online_security",
                    "online_backup",
                    "premium_tech_support",
                    "device_protection_plan",
                    "unlimited_data",
                    "streaming_tv",
                    "streaming_movies",
                    "streaming_music",
                    "satisfaction_score",
                ]
            ]
            # Get overview of data
            st.write(data.head(20))
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button("Predict"):
                # Get batch prediction
                soft_prediction = model.predict_proba(preprocess_df)[:, 1]

                churn_values = []

                for i in range(len(soft_prediction)):
                    if soft_prediction[i] >= model_threshold:
                        churn_value = 1
                        churn_values.append(churn_value)
                    else:
                        churn_value = 0
                        churn_values.append(churn_value)

                if np.sum(churn_values) == 0:
                    st.success("All customers are staying.", icon="✅")
                elif np.sum(churn_values) > 1:
                    st.warning(f"{np.sum(churn_values)} customers are leaving.")
                else:
                    st.warning("1 customer is leaving.")

                prediction_df = pd.DataFrame(
                    {
                        "customer_id": original_data[original_data.columns[0]].tolist(),
                        "churn_probability": soft_prediction,
                        "churn_value": churn_values,
                    }
                )

                prediction_df["recommended_action"] = prediction_df[
                    "churn_value"
                ].apply(
                    lambda x: "Communicate as soon as possible."
                    if x == 1
                    else "Maintain good service."
                )

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader("Prediction")
                st.write(prediction_df)
                # Download results
                results = original_data
                results["churn_probability"] = soft_prediction
                results["threshold"] = model_threshold
                results["churn_value"] = churn_values
                st.download_button(
                    label="Download prediction",
                    data=results.to_csv(index=None),
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
