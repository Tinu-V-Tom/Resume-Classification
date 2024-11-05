import streamlit as st
import pickle
import PyPDF2

# Load the saved classifier and vectorizer
@st.cache_resource
def load_model():
    with open("clf.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return clf, tfidf

clf, tfidf = load_model()

# Define the category mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO/Accountant",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Define the Streamlit app
def main():
    # Add a banner or header image for visual appeal
    st.image("class.jpg", use_column_width=True)  # Add an image named "banner.png" in the same directory

    # Set the title and description
    st.title("Resume Classifier")
    st.markdown("Upload your resume to see how it is classified. Our AI model will analyze the content and categorize it accordingly.")

    # Create tabs for navigation
    tab1, tab2 = st.tabs(["Home", "Add Your Resume to Classify"])

    # Home tab content
    with tab1:
        st.header("Welcome to the Resume Classifier")
        st.markdown("This tool helps classify resumes by analyzing content for relevant job categories.")

    # Resume upload and classification in the second tab
    with tab2:
        st.header("Upload and Classify")

        # Upload PDF file
        uploaded_file = st.file_uploader("Upload a resume PDF", type=["pdf"])

        # Button to classify resume
        if uploaded_file is not None:
            if st.button("Predict"):
                # Extract text from the uploaded PDF file
                resume_text = extract_text_from_pdf(uploaded_file)

                # Transform the text using the vectorizer
                resume_features = tfidf.transform([resume_text])

                # Predict the class
                prediction_id = clf.predict(resume_features)[0]

                # Map prediction_id to the category name
                category_name = category_mapping.get(prediction_id, "Unknown")

                # Display the classification result
                st.subheader("Classification Result")
                st.write(f"Predicted Category: {category_name}")

if __name__ == "__main__":
    main()
