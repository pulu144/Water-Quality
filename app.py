import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
data = pd.read_csv('C:\\Users\\User\\Desktop\\Water Quality\\archive\\water_potability.csv')

# Handling missing values
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Handling class imbalance
ds_low = data[data['Potability'] == 1]
ds_high = data[data['Potability'] == 0]
ds_low_upsampled = resample(ds_low, replace=True, n_samples=len(ds_high), random_state=42)
data = pd.concat([ds_low_upsampled, ds_high])
data = shuffle(data, random_state=42)

# Create a new column for readable labels
data['Potability_Label'] = data['Potability'].map({0: 'Not Potable', 1: 'Potable'})

# Split data into features and target
X = data.drop(['Potability', 'Potability_Label'], axis=1)
y = data['Potability']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train Random Forest Model
rf = RandomForestClassifier(max_depth=20, min_samples_split=2, n_estimators=200, random_state=42)
rf.fit(X_train_pca, y_train)

# Save the model
model_path = 'C:\\Users\\User\\Desktop\\Water Quality\\random_forest_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf, file)

# Streamlit App
st.title("Water Quality Prediction")

# Sidebar Navigation
page = st.sidebar.radio("Choose a page", ["Overview", "Infographic", "Prediction"])

if page == "Overview":
    # Overview Section
    st.header("Overview")
    st.subheader("App Name: Water Quality Prediction")
    st.write("""
    ### Description:
    This project aims to predict the potability of water based on various chemical properties. It leverages several machine learning models and techniques to handle missing data and class imbalance issues.

    ### Issues Addressed:
    - Handling missing values.
    - Balancing class distribution.
    - Visualizing data distributions and relationships.
    - Predicting potability using trained models.
    """)

elif page == "Infographic":
    # Infographic Section
    st.header("Infographic")
    
    # Dataset Details
    st.subheader("Dataset Details")
    st.write(f"Number of samples: {data.shape[0]}")
    st.write(f"Number of attributes: {data.shape[1] - 2}")
    st.write("""
    The dataset consists of water samples analyzed for various chemical properties to determine potability. It includes attributes such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. The goal is to use these attributes to predict whether the water is potable (safe to drink) or not.
    """)

    st.write("""
    #### Attributes:
    - **pH:** Measure of acidity/alkalinity, crucial for maintaining chemical balance and aquatic life.
    - **Hardness:** Measure of the concentration of calcium and magnesium, affecting water quality and household appliances.
    - **Solids:** Total dissolved solids in ppm, indicating the overall mineral content.
    - **Chloramines:** Measure of chlorine compounds in ppm, used for disinfection but harmful in high concentrations.
    - **Sulfate:** Measure of sulfate ion concentration in ppm, which can cause a laxative effect and bad taste in water.
    - **Conductivity:** Measure of water's ability to conduct electricity in µS/cm, indicating ion concentration.
    - **Organic Carbon:** Measure of organic carbon concentration in ppm, affecting taste and odor.
    - **Trihalomethanes:** Measure of trihalomethanes concentration in µg/L, a byproduct of chlorination that is harmful at high levels.
    - **Turbidity:** Measure of water clarity in NTU, indicating the presence of suspended particles that can harbor pathogens.
    """)

    # Water Quality Ratio
    st.subheader("Water Quality Ratio")
    st.write("This pie chart shows the proportion of potable vs non-potable water samples, giving a quick visual of the balance between the two categories in the dataset.")
    fig, ax = plt.subplots()
    data['Potability_Label'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

    # Distribution of Quality in Both Water Types
    st.subheader("Distribution of Quality in Both Water Types")
    st.write("This bar chart shows the distribution of potable and non-potable water samples, providing a clear view of the class imbalance in the dataset.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x='Potability_Label', ax=ax)
    st.pyplot(fig)

    # Spread of Nutrients in Water
    st.subheader("Spread of Nutrients in Water")
    st.write("This boxplot shows the distribution of different nutrients in the water samples. It helps identify the range, median, and potential outliers for each nutrient.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data.drop(['Potability', 'Potability_Label'], axis=1), ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Distribution of Each Nutrient
    st.subheader("Distribution of Each Nutrient")
    st.write("These histograms show the distribution of each nutrient in the water samples. Each plot provides insights into the central tendency, spread, and skewness of the nutrient levels.")
    nutrients = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for ax, nutrient in zip(axes, nutrients):
        sns.histplot(data[nutrient], kde=True, ax=ax)
        ax.set_title(f'{nutrient} Distribution')
    st.pyplot(fig)
    
    # Scatter Plot
    st.subheader("pH and Hardness")
    st.write("This scatter plot shows the relationship between pH and Hardness of the water samples, colored by potability. It helps visualize any correlation between these two variables and how they relate to potability.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='ph', y='Hardness', hue='Potability_Label', ax=ax)
    st.pyplot(fig)

    # Histogram Plot
    st.subheader("Conductivity and Count")
    st.write("This histogram shows the distribution of water conductivity, colored by potability. It provides a clear visual of how conductivity levels vary between potable and non-potable water samples.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='Conductivity', hue='Potability_Label', multiple='stack', ax=ax)
    st.pyplot(fig)

elif page == "Prediction":
    # Prediction Section
    st.header("Prediction")
    st.subheader("Enter the values for the following attributes:")

    # Creating input fields for user to input data
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
    Hardness = st.number_input('Hardness', min_value=0.0, value=200.0)
    Solids = st.number_input('Solids', min_value=0.0, value=20000.0)
    Chloramines = st.number_input('Chloramines', min_value=0.0, value=7.0)
    Sulfate = st.number_input('Sulfate', min_value=0.0, value=300.0)
    Conductivity = st.number_input('Conductivity', min_value=0.0, value=500.0)
    Organic_carbon = st.number_input('Organic Carbon', min_value=0.0, value=15.0)
    Trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=70.0)
    Turbidity = st.number_input('Turbidity', min_value=0.0, value=4.0)

    # When 'Predict' button is pressed
    if st.button('Predict'):
        input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        input_data_scaled = scaler.transform(input_data)
        input_data_pca = pca.transform(input_data_scaled)
        
        prediction = rf.predict(input_data_pca)
        
        if prediction[0] == 1:
            st.write("The water is potable.")
        else:
            st.write("The water is not potable.")
