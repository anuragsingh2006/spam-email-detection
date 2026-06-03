import streamlit as st
import pickle

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E88E5;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    '<p class="main-title">📧 Spam Email Detector</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Detect whether an email is Spam or Ham using Machine Learning</p>',
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This application uses a Machine Learning model
        to classify emails as:
        
        ✅ Ham (Legitimate Email)
        
        🚨 Spam (Unwanted Email)
        """
    )

    st.header("📌 Sample Email")
    st.info(
        "Congratulations! You've won a free iPhone. "
        "Click here to claim your reward."
    )

# ----------------------------
# Input Section
# ----------------------------
st.subheader("✍️ Enter Email Content")

email = st.text_area(
    "",
    height=200,
    placeholder="Paste your email text here..."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict", use_container_width=True):

    if email.strip() == "":
        st.warning("Please enter some email text.")
    else:

        with st.spinner("Analyzing email..."):

            transformed_email = vectorizer.transform([email])

            prediction = model.predict(transformed_email)

            # Probability (if model supports it)
            try:
                probability = model.predict_proba(
                    transformed_email
                )[0]

                confidence = max(probability) * 100

            except:
                confidence = None

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:

            st.error("🚨 This Email is SPAM")

            if confidence:
                st.progress(int(confidence))
                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )

        else:

            st.success("✅ This Email is NOT SPAM")

            if confidence:
                st.progress(int(confidence))
                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )

# ----------------------------
# Footer
# ----------------------------
st.divider()

st.markdown(
    """
    <center>
    Built with ❤️ using Streamlit & Machine Learning
    </center>
    """,
    unsafe_allow_html=True
)