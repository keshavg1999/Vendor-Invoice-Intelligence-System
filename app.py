import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# CUSTOM STYLING
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #5f6b76;
        margin-bottom: 1.5rem;
    }

    .section-header {
        color: #1f4e79;
        font-size: 1.6rem;
        font-weight: 600;
    }

    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f4f7fb;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">📦 Vendor Invoice Intelligence Portal</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Driven Freight Cost Prediction & Invoice Risk Flagging
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    This internal analytics portal uses machine learning to support
    vendor invoice analysis, freight forecasting, and manual approval
    decisions.
    """
)

st.divider()


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("🔍 Prediction Modules")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("📊 Business Impact")

st.sidebar.markdown(
    """
    - 📉 Improved freight cost forecasting
    - 🧾 Faster invoice screening
    - 🚨 Identification of invoices requiring review
    - ⚙️ More efficient finance operations
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine learning predictions should be reviewed according "
    "to your organization's approval policies."
)


# ================================================================
# FREIGHT COST PREDICTION
# ================================================================

if selected_model == "Freight Cost Prediction":

    st.markdown(
        '<div class="section-header">🚚 Freight Cost Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **Objective:** Predict the expected freight cost for a vendor
        invoice using **Quantity** and **Invoice Dollars**.
        """
    )

    st.info(
        "Enter the invoice quantity and invoice dollar amount, "
        "then click **Predict Freight Cost**."
    )

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:

            quantity = st.number_input(
                "📦 Quantity",
                min_value=1,
                value=1200,
                step=1,
                help="Number of items on the invoice."
            )

        with col2:

            dollars = st.number_input(
                "💵 Invoice Dollars",
                min_value=0.01,
                value=18500.00,
                step=100.00,
                format="%.2f",
                help="Total dollar value of the invoice."
            )

        submit_freight = st.form_submit_button(
            "🔮 Predict Freight Cost",
            use_container_width=True
        )

    if submit_freight:

        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        try:

            prediction = predict_freight_cost(
                input_data
            )

            predicted_freight = float(
                prediction["Predicted_Freight"].iloc[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="📦 Quantity",
                    value=f"{quantity:,}"
                )

            with col2:
                st.metric(
                    label="💵 Invoice Value",
                    value=f"${dollars:,.2f}"
                )

            with col3:
                st.metric(
                    label="🚚 Estimated Freight",
                    value=f"${predicted_freight:,.2f}"
                )

            st.divider()

            st.subheader("Prediction Details")

            result_data = prediction.copy()

            result_data["Quantity"] = result_data[
                "Quantity"
            ].map(lambda x: f"{x:,.0f}")

            result_data["Dollars"] = result_data[
                "Dollars"
            ].map(lambda x: f"${x:,.2f}")

            result_data["Predicted_Freight"] = result_data[
                "Predicted_Freight"
            ].map(lambda x: f"${x:,.2f}")

            st.dataframe(
                result_data,
                use_container_width=True,
                hide_index=True
            )

        except FileNotFoundError:

            st.error(
                "❌ Freight prediction model was not found. "
                "Please train the model first."
            )

        except Exception as e:

            st.error(
                f"❌ Unable to generate freight prediction: {e}"
            )


# ================================================================
# INVOICE MANUAL APPROVAL FLAG
# ================================================================

else:

    st.markdown(
        '<div class="section-header">🚨 Invoice Manual Approval Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **Objective:** Evaluate an invoice using quantity, dollar value,
        freight, and item-level totals to determine whether it should
        be flagged for manual review.
        """
    )

    st.warning(
        "This prediction is a machine-learning risk indicator and "
        "does not replace your organization's approval policy."
    )

    with st.form("invoice_flag_form"):

        st.subheader("Invoice Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "📦 Invoice Quantity",
                min_value=1,
                value=50,
                step=1
            )

            invoice_dollars = st.number_input(
                "💵 Invoice Dollars",
                min_value=0.01,
                value=352.95,
                step=10.00,
                format="%.2f"
            )

        with col2:

            freight = st.number_input(
                "🚚 Freight Cost",
                min_value=0.0,
                value=1.73,
                step=0.10,
                format="%.2f"
            )

            total_item_quantity = st.number_input(
                "📦 Total Item Quantity",
                min_value=1,
                value=162,
                step=1
            )

        with col3:

            total_item_dollars = st.number_input(
                "💰 Total Item Dollars",
                min_value=0.01,
                value=2476.00,
                step=10.00,
                format="%.2f"
            )

        submit_flag = st.form_submit_button(
            "🧠 Evaluate Invoice Risk",
            use_container_width=True
        )

    if submit_flag:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        try:

            prediction = predict_invoice_flag(
                input_data
            )

            flag_prediction = prediction[
                "Predicted_Flag"
            ].iloc[0]

            is_flagged = bool(flag_prediction)

            st.divider()

            # ----------------------------------------------------
            # Prediction Result
            # ----------------------------------------------------

            if is_flagged:

                st.error(
                    "🚨 Invoice requires **MANUAL APPROVAL**"
                )

                st.warning(
                    "The model has flagged this invoice for "
                    "additional review."
                )

                status = "MANUAL APPROVAL"

            else:

                st.success(
                    "✅ Invoice is **SAFE for AUTO-APPROVAL**"
                )

                st.info(
                    "The model did not flag this invoice based "
                    "on the supplied inputs."
                )

                status = "AUTO-APPROVAL"

            # ----------------------------------------------------
            # Invoice Summary
            # ----------------------------------------------------

            st.subheader("Invoice Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Invoice Quantity",
                    f"{invoice_quantity:,}"
                )

            with col2:
                st.metric(
                    "Invoice Value",
                    f"${invoice_dollars:,.2f}"
                )

            with col3:
                st.metric(
                    "Freight",
                    f"${freight:,.2f}"
                )

            with col4:
                st.metric(
                    "Decision",
                    status
                )

            st.subheader("Prediction Details")

            result_data = prediction.copy()

            st.dataframe(
                result_data,
                use_container_width=True,
                hide_index=True
            )

        except FileNotFoundError:

            st.error(
                "❌ Invoice flag prediction model was not found. "
                "Please train the invoice flag model first."
            )

        except Exception as e:

            st.error(
                f"❌ Unable to evaluate invoice: {e}"
            )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "Vendor Invoice Intelligence Portal | "
    "Machine Learning Decision Support System"
)
