import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
st.set_option('deprecation.showPyplotGlobalUse', False)

# === PART 1: Data Generation ===
@st.cache_data
def generate_data(num_samples=10000, filename='personal_wellness_dataset_large.csv'):
    """Generates synthetic wellness data with realistic correlations."""
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df

    data = []
    stress_levels = ['Low', 'Moderate', 'High']
    moods = ['Happy', 'Sad', 'Neutral', 'Energetic', 'Tired']
    hydration_levels = ['Low', 'Medium', 'High']
    mental_focus_levels = ['Low', 'Medium', 'High']
    bowel_movement_frequency = ['Daily', 'Every Other Day', 'Twice Daily', 'Irregular']
    energy_levels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    water_intake_ml_options = [1500, 2000, 2500, 3000, 3500]
    steps_taken_options = [3000, 5000, 7000, 10000, 12000]
    heart_rate_resting_options = [55, 60, 65, 70, 75]
    systolic_bp_options = [110, 120, 130]
    diastolic_bp_options = [70, 80, 90]
    cholesterol_levels_options = ['Normal', 'Borderline High', 'High']
    blood_sugar_levels_options = ['Normal', 'Prediabetes', 'Diabetes']
    recommended_water_ml_options = [2000, 2500, 3000]
    suggested_meal_timing_options = ['Regular Intervals', 'Intermittent Fasting', 'Time-Restricted Eating']
    recommended_supplements_options = [['Vitamin D', 'Omega-3'], ['Magnesium', 'Zinc'], ['Probiotics'], [], ['Multivitamin']]
    suggested_mindfulness_activity_options = ['Meditation', 'Deep Breathing', 'Yoga', 'Nature Walk', 'Reading']
    sleep_efficiency_percentage_options = [70, 80, 90, 95]
    meal_frequency_options = ['3 Meals a Day', '5-6 Smaller Meals', '2-3 Larger Meals']
    wellness_score_options = ['Poor', 'Average', 'Good', 'Excellent']

    for _ in range(num_samples):
        age = np.random.randint(18, 60)
        bmi = round(np.random.normal(25, 3), 1)
        objective = random.choice(['Weight Gain', 'Weight Loss', 'Maintenance'])
        timeline = random.choice(['1 Month', '3 Months', '6 Months'])
        diet_type = random.choice(['Vegan', 'Vegetarian', 'Keto', 'High Protein', 'Balanced'])
        workout_hours = round(np.random.uniform(0.3, 2.5), 1)
        workout_type = random.choice(['Cardio', 'Strength', 'Yoga', 'HIIT', 'Rest'])
        activity_level_options = ['Sedentary', 'Lightly Active', 'Active', 'Very Active']
        
        if bmi > 30:
            activity_level = random.choices(activity_level_options, weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
        elif bmi < 20:
            activity_level = random.choices(activity_level_options, weights=[0.1, 0.2, 0.3, 0.4], k=1)[0]
        else:
            activity_level = random.choice(activity_level_options)

        if activity_level == 'Sedentary':
            daily_steps = random.choice([1500, 3000, 4000])
        elif activity_level == 'Lightly Active':
            daily_steps = random.choice([4000, 5000, 6000, 7000])
        elif activity_level == 'Active':
            daily_steps = random.choice([7000, 8000, 9000, 10000, 11000])
        else:
            daily_steps = random.choice([10000, 12000, 14000, 16000])

        stress_level = random.choice(stress_levels)
        if stress_level == 'High':
            mood = random.choices(moods, weights=[0.2, 0.4, 0.3, 0.05, 0.05], k=1)[0]
        elif stress_level == 'Low':
            mood = random.choices(moods, weights=[0.5, 0.1, 0.2, 0.1, 0.1], k=1)[0]
        else:
            mood = random.choice(moods)

        water_intake_ml = random.choice(water_intake_ml_options)
        if water_intake_ml < 2000:
            hydration_level = random.choices(hydration_levels, weights=[0.6, 0.3, 0.1], k=1)[0]
        elif water_intake_ml < 3000:
            hydration_level = random.choices(hydration_levels, weights=[0.1, 0.6, 0.3], k=1)[0]
        else:
            hydration_level = random.choices(hydration_levels, weights=[0.05, 0.3, 0.65], k=1)[0]

        mental_focus = random.choice(mental_focus_levels)
        bowel_movement = random.choice(bowel_movement_frequency)

        resting_heart_rate_base = 70
        resting_heart_rate_adjustment = 0
        if age > 45:
            resting_heart_rate_adjustment += 3
        if bmi > 28:
            resting_heart_rate_adjustment += 2
        elif bmi < 20:
            resting_heart_rate_adjustment -= 2
        resting_heart_rate = np.clip(random.choice(heart_rate_resting_options) + resting_heart_rate_adjustment, 50, 85)

        systolic_pressure_base = 120
        diastolic_pressure_base = 80
        bp_adjustment_systolic = 0
        bp_adjustment_diastolic = 0
        if age > 50:
            bp_adjustment_systolic += 5
            bp_adjustment_diastolic += 3
        if bmi > 30:
            bp_adjustment_systolic += 8
            bp_adjustment_diastolic += 5
        systolic_pressure = np.clip(random.choice(systolic_bp_options) + bp_adjustment_systolic, 100, 150)
        diastolic_pressure = np.clip(random.choice(diastolic_bp_options) + bp_adjustment_diastolic, 60, 100)

        if diet_type == 'Keto' or bmi > 30:
            cholesterol = random.choices(cholesterol_levels_options, weights=[0.5, 0.3, 0.2], k=1)[0]
            blood_sugar = random.choices(blood_sugar_levels_options, weights=[0.6, 0.3, 0.1], k=1)[0]
        elif diet_type == 'High Protein':
            cholesterol = random.choices(cholesterol_levels_options, weights=[0.6, 0.3, 0.1], k=1)[0]
            blood_sugar = random.choice(blood_sugar_levels_options)
        elif diet_type == 'Vegan':
            cholesterol = random.choices(cholesterol_levels_options, weights=[0.8, 0.15, 0.05], k=1)[0]
            blood_sugar = random.choice(blood_sugar_levels_options)
        else:
            cholesterol = random.choice(cholesterol_levels_options)
            blood_sugar = random.choice(blood_sugar_levels_options)

        recommended_calories = np.random.randint(1800, 3000)
        protein_g = np.random.randint(50, 200)
        carbs_g = np.random.randint(150, 350)
        fats_g = np.random.randint(40, 100)
        vitamins = random.sample(['A', 'B12', 'C', 'D', 'E', 'K', 'Folate'], 3)

        sleep_duration = round(np.random.normal(7, 1), 1)
        wake_up_time_options = ['05:30', '06:00', '06:30', '07:00', '07:30', '08:00']
        if sleep_duration < 6:
            sleep_quality = random.choices(['Poor', 'Average', 'Good'], weights=[0.6, 0.3, 0.1], k=1)[0]
            perceived_energy_level = random.choices(energy_levels, weights=[0.6, 0.3, 0.1, 0.0, 0.0], k=1)[0]
            new_sleep_duration = round(np.clip(sleep_duration + np.random.normal(0.5, 0.3), 6, 8), 1)
            new_wake_up_time = random.choice(['06:30', '07:00', '07:30', '08:00'])
        elif sleep_duration > 8:
            sleep_quality = random.choices(['Average', 'Good', 'Excellent'], weights=[0.2, 0.5, 0.3], k=1)[0]
            perceived_energy_level = random.choices(energy_levels, weights=[0.0, 0.1, 0.3, 0.4, 0.2], k=1)[0]
            new_sleep_duration = round(np.clip(sleep_duration + np.random.normal(-0.3, 0.5), 7, 9), 1)
            new_wake_up_time = random.choice(['05:30', '06:00', '06:30', '07:00'])
        else:
            sleep_quality = random.choice(['Poor', 'Average', 'Good', 'Excellent'])
            perceived_energy_level = random.choice(energy_levels)
            new_sleep_duration = round(np.clip(sleep_duration + np.random.normal(0, 0.6), 6.5, 8.5), 1)
            new_wake_up_time = random.choice(['06:00', '06:30', '07:00', '07:15'])
        wake_up_time = random.choice(wake_up_time_options)

        suggested_workout = random.choice(['Cardio + Strength', 'HIIT + Yoga', 'Strength Focused', 'Light Cardio', 'Rest Day'])
        suggested_water_intake = random.choice(recommended_water_ml_options)
        meal_timing_advice = random.choice(suggested_meal_timing_options)
        suggested_supplements_list = random.choice(recommended_supplements_options)
        suggested_mindfulness = random.choice(suggested_mindfulness_activity_options)
        sleep_efficiency = random.choice(sleep_efficiency_percentage_options)
        recommended_workout_duration = round(np.clip(workout_hours + np.random.normal(0.2, 0.5), 0.5, 2.5), 1)
        recommended_meal_frequency = random.choice(meal_frequency_options)

        if activity_level in ['Sedentary', 'Lightly Active']:
            target_daily_steps = random.choice([7000, 8000, 9000])
        else:
            target_daily_steps = random.choice([10000, 11000, 12000, 13000])

        wellness_factors = [bmi >= 18.5 and bmi <= 24.9, sleep_duration >= 7 and sleep_duration <= 9,
                            stress_level == 'Low' or stress_level == 'Moderate', mood in ['Happy', 'Energetic'],
                            hydration_level == 'High' or hydration_level == 'Medium', daily_steps >= 7000,
                            resting_heart_rate >= 55 and resting_heart_rate <= 75,
                            systolic_pressure >= 110 and systolic_pressure <= 130,
                            diastolic_pressure >= 70 and diastolic_pressure <= 90,
                            cholesterol_levels_options.index(cholesterol) < 2,
                            blood_sugar_levels_options.index(blood_sugar) < 2]

        positive_factors = sum(wellness_factors)
        if positive_factors >= 9:
            overall_wellness_score = 'Excellent'
        elif positive_factors >= 7:
            overall_wellness_score = 'Good'
        elif positive_factors >= 5:
            overall_wellness_score = 'Average'
        else:
            overall_wellness_score = 'Poor'

        data.append({
            'Age': age, 'BMI': bmi, 'Objective': objective, 'Timeline': timeline, 'Diet Type': diet_type,
            'Sleep Duration (hrs)': sleep_duration, 'Wake Up Time': wake_up_time, 'Sleep Quality': sleep_quality,
            'Workout Duration (hrs)': workout_hours, 'Workout Type': workout_type, 'Activity Level': activity_level,
            'Stress Level': stress_level, 'Mood': mood, 'Hydration Level': hydration_level,
            'Mental Focus': mental_focus, 'Bowel Movement Frequency': bowel_movement, 'Daily Steps': daily_steps,
            'Resting Heart Rate': resting_heart_rate, 'Systolic Blood Pressure': systolic_pressure,
            'Diastolic Blood Pressure': diastolic_pressure,
            'Recommended Calories': recommended_calories, 'Protein (g)': protein_g, 'Carbs (g)': carbs_g, 'Fats (g)': fats_g,
            'Vitamins': ', '.join(vitamins), 'New Sleep Duration (hrs)': new_sleep_duration, 'New Wake Time': new_wake_up_time,
            'Suggested Workout': suggested_workout, 'Recommended Water Intake (ml)': suggested_water_intake,
            'Meal Timing Advice': meal_timing_advice,
            'Suggested Supplements': ', '.join(suggested_supplements_list) if suggested_supplements_list else None,
            'Suggested Mindfulness Activity': suggested_mindfulness, 'Perceived Energy Level': perceived_energy_level,
            'Sleep Efficiency (%)': sleep_efficiency, 'Cholesterol Level': cholesterol, 'Blood Sugar Level': blood_sugar,
            'Recommended Workout Duration (hrs)': recommended_workout_duration,
            'Recommended Meal Frequency': recommended_meal_frequency,
            'Target Daily Steps': target_daily_steps,
            'Overall Wellness Score': overall_wellness_score
        })

    df = pd.DataFrame(data)
    df['New Sleep Duration (hrs)'] = df['New Sleep Duration (hrs)'].astype(float)
    df['Recommended Workout Duration (hrs)'] = df['Recommended Workout Duration (hrs)'].astype(float)
    df['Target Daily Steps'] = df['Target Daily Steps'].astype(int)
    df.to_csv(filename, index=False)
    return df

# === PART 2: Machine Learning Model ===
@st.cache_resource
def train_models(df):
    """Trains and returns the machine learning models."""
    input_columns = [
        'Age', 'BMI', 'Objective', 'Timeline', 'Diet Type',
        'Sleep Duration (hrs)', 'Wake Up Time', 'Sleep Quality',
        'Workout Duration (hrs)', 'Workout Type', 'Activity Level',
        'Stress Level', 'Mood', 'Hydration Level', 'Mental Focus',
        'Bowel Movement Frequency', 'Daily Steps', 'Resting Heart Rate',
        'Systolic Blood Pressure', 'Diastolic Blood Pressure'
    ]
    
    output_columns_reg = [
        'Recommended Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)',
        'New Sleep Duration (hrs)', 'Recommended Workout Duration (hrs)',
        'Target Daily Steps'
    ]
    
    output_columns_clf = [
        'Suggested Workout', 'Meal Timing Advice',
        'Recommended Meal Frequency', 'Overall Wellness Score'
    ]
    
    df = df.dropna(subset=output_columns_reg + output_columns_clf)
    df[output_columns_clf] = df[output_columns_clf].astype(str)
    
    X = df[input_columns]
    y_reg = df[output_columns_reg]
    y_clf = df[output_columns_clf]
    
    # Time conversion function
    def time_to_minutes(time_str):
        try:
            time_str = str(time_str).split('.')[0]
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except Exception:
            return np.nan
    
    X = X.copy()
    X['Wake Up Time Minutes'] = X['Wake Up Time'].apply(time_to_minutes)
    median_wake_time = X['Wake Up Time Minutes'].median()
    X['Wake Up Time Minutes'].fillna(median_wake_time, inplace=True)
    X = X.drop('Wake Up Time', axis=1)
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    multi_output_regressor = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('multioutput_reg', MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42, 
                                                                     n_jobs=-1, max_depth=20, min_samples_split=8, 
                                                                     min_samples_leaf=4)))
    ])
    
    classifier_pipelines = {}
    for target_col in output_columns_clf:
        classifier_pipelines[target_col] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, 
                                                max_depth=20, min_samples_split=8, min_samples_leaf=4, 
                                                class_weight='balanced'))
        ])
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.25, random_state=42
    )
    
    multi_output_regressor.fit(X_train, y_reg_train)
    
    for target_col, pipeline in classifier_pipelines.items():
        current_y_clf_train = y_clf_train[target_col]
        pipeline.fit(X_train, current_y_clf_train)
    
    return {
        'reg_model': multi_output_regressor,
        'clf_models': classifier_pipelines,
        'X_train': X_train,
        'input_columns': input_columns,
        'output_columns_reg': output_columns_reg,
        'output_columns_clf': output_columns_clf
    }

# === UI Functions ===
def plot_input_vs_training_distribution(user_input_dict, training_data):
    """Plots user input against training data distributions."""
    num_cols = ['Age', 'BMI', 'Sleep Duration (hrs)', 'Workout Duration (hrs)', 
               'Daily Steps', 'Resting Heart Rate', 'Wake Up Time Minutes']
    cat_cols = ['Objective', 'Diet Type', 'Sleep Quality', 'Workout Type', 
               'Activity Level', 'Stress Level']
    
    num_plots = len(num_cols) + len(cat_cols)
    if num_plots == 0:
        return
    
    cols_per_row = 3
    num_rows = (num_plots + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    axes = axes.flatten()
    
    plot_index = 0
    
    # Numerical Plots
    for col in num_cols:
        if col not in user_input_dict or col not in training_data.columns:
            continue
        ax = axes[plot_index]
        sns.histplot(training_data[col], kde=True, color='skyblue', stat='density', ax=ax)
        try:
            user_val = float(user_input_dict[col])
            ax.axvline(user_val, color='red', linestyle='--', linewidth=2, label=f'Your Input ({user_val:.1f})')
            ax.legend()
        except (ValueError, TypeError):
            pass
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        plot_index += 1
    
    # Categorical Plots
    for col in cat_cols:
        if col not in user_input_dict or col not in training_data.columns:
            continue
        ax = axes[plot_index]
        order = training_data[col].value_counts().index
        sns.countplot(y=training_data[col], order=order, palette='viridis', ax=ax)
        user_choice = user_input_dict.get(col, 'N/A')
        ax.set_title(f'Distribution of {col}\n(Your Choice: {user_choice})')
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
        plot_index += 1
    
    # Hide any unused axes
    for i in range(plot_index, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

def prepare_input_data(input_dict, X_train):
    """Convert user input dictionary to DataFrame and apply time conversion."""
    input_df = pd.DataFrame([input_dict])
    
    # Time conversion function
    def time_to_minutes(time_str):
        try:
            time_str = str(time_str).split('.')[0]
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except Exception:
            return np.nan
    
    input_df['Wake Up Time Minutes'] = input_df['Wake Up Time'].apply(time_to_minutes)
    
    if input_df['Wake Up Time Minutes'].isnull().any():
        median_wake_time = X_train['Wake Up Time Minutes'].median()
        input_df['Wake Up Time Minutes'].fillna(median_wake_time, inplace=True)
    
    input_df = input_df.drop('Wake Up Time', axis=1)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    
    return input_df

def make_predictions(input_data_dict, models, X_train):
    """Make predictions using the trained models."""
    processed_input_df = prepare_input_data(input_data_dict, X_train)
    
    all_predictions = {}
    
    # Regression predictions
    reg_predictions_array = models['reg_model'].predict(processed_input_df)
    for i, col in enumerate(models['output_columns_reg']):
        pred_val = reg_predictions_array[0, i]
        if col == 'Recommended Calories':
            all_predictions[col] = int(round(pred_val))
        elif col == 'New Sleep Duration (hrs)':
            all_predictions[col] = round(pred_val, 1)
        elif col == 'Recommended Workout Duration (hrs)':
            all_predictions[col] = round(pred_val, 1)
        elif col == 'Target Daily Steps':
            all_predictions[col] = int(round(pred_val))
        else:
            all_predictions[col] = round(pred_val, 1)
    
    # Classification predictions
    for target_col, pipeline in models['clf_models'].items():
        clf_pred = pipeline.predict(processed_input_df)[0]
        all_predictions[target_col] = clf_pred
    
    return all_predictions

def get_user_input():
    """Get user input for prediction using Streamlit form."""
    st.subheader("Personal Health Information")
    
    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 60, 35)
            bmi = st.slider("BMI", 15.0, 40.0, 22.5, step=0.1)
            objective = st.selectbox("Objective", ["Weight Gain", "Weight Loss", "Maintenance"])
            timeline = st.selectbox("Timeline", ["1 Month", "3 Months", "6 Months"])
            diet_type = st.selectbox("Diet Type", ["Vegan", "Vegetarian", "Keto", "High Protein", "Balanced"])
            sleep_duration = st.slider("Sleep Duration (hrs)", 4.0, 12.0, 7.5, step=0.1)
            wake_up_time = st.text_input("Wake Up Time (HH:MM)", "06:30")
            sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
        
        with col2:
            workout_hours = st.slider("Workout Duration (hrs)", 0.0, 4.0, 1.0, step=0.1)
            workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "Yoga", "HIIT", "Rest"])
            activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Active", "Very Active"])
            stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
            mood = st.selectbox("Mood", ["Happy", "Sad", "Neutral", "Energetic", "Tired"])
            hydration_level = st.selectbox("Hydration Level", ["Low", "Medium", "High"])
            mental_focus = st.selectbox("Mental Focus", ["Low", "Medium", "High"])
            bowel_movement = st.selectbox("Bowel Movement Frequency", ["Daily", "Every Other Day", "Twice Daily", "Irregular"])
        
        daily_steps = st.slider("Daily Steps", 1000, 20000, 10000, step=1000)
        resting_heart_rate = st.slider("Resting Heart Rate", 50, 100, 65)
        
        col3, col4 = st.columns(2)
        with col3:
            systolic_pressure = st.slider("Systolic Blood Pressure", 90, 160, 120)
        with col4:
            diastolic_pressure = st.slider("Diastolic Blood Pressure", 60, 100, 80)
        
        submitted = st.form_submit_button("Get Wellness Recommendations")
        
        if submitted:
            user_data = {
                'Age': age, 'BMI': bmi, 'Objective': objective, 'Timeline': timeline, 
                'Diet Type': diet_type, 'Sleep Duration (hrs)': sleep_duration, 
                'Wake Up Time': wake_up_time, 'Sleep Quality': sleep_quality,
                'Workout Duration (hrs)': workout_hours, 'Workout Type': workout_type, 
                'Activity Level': activity_level, 'Stress Level': stress_level, 'Mood': mood,
                'Hydration Level': hydration_level, 'Mental Focus': mental_focus, 
                'Bowel Movement Frequency': bowel_movement, 'Daily Steps': daily_steps,
                'Resting Heart Rate': resting_heart_rate, 
                'Systolic Blood Pressure': systolic_pressure, 
                'Diastolic Blood Pressure': diastolic_pressure
            }
            
            # Calculate wake up time in minutes for visualization
            try:
                hours, minutes = map(int, wake_up_time.split(':'))
                user_data['Wake Up Time Minutes'] = hours * 60 + minutes
            except:
                user_data['Wake Up Time Minutes'] = 390  # Default to 6:30 (390 minutes)
            
            return user_data
    
    return None

# === Main Streamlit App ===
def main():
    # Configure page
    st.set_page_config(
        page_title="Wellness Prediction System", 
        page_icon="🌿", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
            .stApp { background-color: #f0f5f5; }
            .stMarkdown h1 { color: #2e8b57; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stAlert { border-left: 5px solid #4CAF50; }
            .css-1aumxhk { background-color: #e8f5e9; border-radius: 10px; padding: 20px; }
            .css-1kyxreq { margin-top: 20px; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🌿 Personalized Wellness Prediction System")
    st.markdown("""
        Welcome to the Personalized Wellness Prediction System! This AI-powered tool analyzes your health metrics 
        to provide personalized recommendations for nutrition, exercise, sleep, and overall wellness.
    """)
    
    # Generate data and train models
    with st.spinner("Loading wellness data and training models..."):
        df = generate_data(num_samples=10000)
        models = train_models(df)
    
    # Get user input
    user_data = get_user_input()
    
    if user_data:
        st.success("✅ Your health information has been received!")
        
        # Show input summary
        st.subheader("Your Health Profile")
        input_df = pd.DataFrame([user_data]).drop('Wake Up Time Minutes', axis=1)
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        
        # Visualizations
        st.subheader("Your Profile Compared to Our Dataset")
        plot_input_vs_training_distribution(user_data, models['X_train'])
        
        # Make predictions
        with st.spinner("Generating personalized wellness recommendations..."):
            predictions = make_predictions(user_data, models, models['X_train'])
        
        # Display predictions
        st.subheader("Your Personalized Wellness Recommendations")
        
        # Wellness score
        wellness_score = predictions['Overall Wellness Score']
        st.markdown(f"### 🌟 Overall Wellness Score: **{wellness_score}**")
        
        # Nutrition recommendations
        st.markdown("### 🍎 Nutrition Recommendations")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recommended Calories", predictions['Recommended Calories'])
        col2.metric("Protein (g)", predictions['Protein (g)'])
        col3.metric("Carbs (g)", predictions['Carbs (g)'])
        col4.metric("Fats (g)", predictions['Fats (g)'])
        
        st.markdown(f"**Meal Timing Advice:** {predictions['Meal Timing Advice']}")
        st.markdown(f"**Recommended Meal Frequency:** {predictions['Recommended Meal Frequency']}")
        
        # Exercise recommendations
        st.markdown("### 💪 Exercise Recommendations")
        col5, col6 = st.columns(2)
        col5.metric("Recommended Workout Type", predictions['Suggested Workout'])
        col6.metric("Recommended Workout Duration (hrs)", predictions['Recommended Workout Duration (hrs)'])
        st.markdown(f"**Target Daily Steps:** {predictions['Target Daily Steps']:,}")
        
        # Sleep recommendations
        st.markdown("### 😴 Sleep Recommendations")
        col7, col8 = st.columns(2)
        col7.metric("Recommended Sleep Duration (hrs)", predictions['New Sleep Duration (hrs)'])
        col8.metric("Wake Up Time", user_data.get('New Wake Time', '06:30'))
        
        # Mindfulness recommendations
        st.markdown("### 🧘 Mindfulness & Wellness")
        st.markdown(f"**Suggested Mindfulness Activity:** {predictions['Suggested Mindfulness Activity']}")
        
        # Supplements
        if predictions.get('Suggested Supplements'):
            st.markdown(f"**Suggested Supplements:** {predictions['Suggested Supplements']}")
        else:
            st.markdown("**Suggested Supplements:** No supplements recommended at this time")
        
        st.markdown("---")
        st.info("💡 Remember that these are AI-generated recommendations. Always consult with a healthcare professional before making significant changes to your wellness routine.")
    
    # Sidebar information
    with st.sidebar:
        st.header("About This System")
        st.markdown("""
            This wellness prediction system uses machine learning to analyze:
            - Personal health metrics
            - Lifestyle factors
            - Wellness goals
            
            The system provides personalized recommendations for:
            - Nutrition plans
            - Exercise routines
            - Sleep optimization
            - Stress management
            - Overall wellness improvement
        """)
        
        st.markdown("---")
        st.subheader("How It Works")
        st.markdown("""
            1. Enter your health information
            2. Our AI analyzes your profile
            3. Receive personalized recommendations
            4. Implement suggestions for better health
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Python, Scikit-learn, and Streamlit")

if __name__ == "__main__":
    main()