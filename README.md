# Diabetes Prediction Web Application
*A complete machine learning project ready for deployment*  
**Author: Hemanth Peddada**  
**Date: July 2025**

## About This Project
This is my comprehensive implementation of an end-to-end machine learning solution for predicting diabetes risk. I built this project to demonstrate skills in data science, machine learning, and web development. The application uses the Pima Indians Diabetes Dataset to provide diabetes risk assessments through a web interface.

## 🚀 Quick Start
To run the application:
1. Clone the repository
2. Install dependencies: `pip install -r deployment/requirements.txt`
3. Run the application: `python webapp/app.py`
4. Access the web interface to make predictions

## 📊 Project Highlights
- **Machine Learning Model**: Diabetes risk prediction algorithm
- **Interactive Web Interface**: User-friendly prediction form
- **Data Processing**: Complete dataset analysis and preprocessing
- **API Integration**: RESTful endpoints for predictions
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technical Features
- **Backend**: Python Flask with advanced routing
- **Frontend**: HTML5, CSS3, Bootstrap for responsive design
- **Machine Learning**: Custom prediction model with validation
- **APIs**: REST endpoints for programmatic access
- **Error Handling**: Comprehensive input validation and error management
- **Logging**: Application monitoring and debugging capabilities

##  Key Features

### Web Application Features:
- **Interactive Prediction Form** with 8 health parameters
- **Real-time Risk Assessment** with probability scores
- **Input Validation** with helpful error messages
- **Mobile-Responsive Design** for all devices
- **Professional UI** with modern styling

### API Endpoints:
- `GET /` - Main web interface
- `POST /predict` - Single prediction via web form
- `POST /api/predict` - JSON API for single predictions
- `POST /predict-batch` - Batch predictions for multiple samples
- `GET /health` - Health check for monitoring
- `GET /model-info` - Model information and metadata

### Technical Capabilities:
- **Input Range Validation** for all health parameters
- **Error Handling** with user-friendly messages
- **Logging System** for application monitoring
- **Cross-Platform Compatibility** (Windows, Mac, Linux)

## 💻 Running the Application

### Prerequisites
```bash
pip install -r deployment/requirements.txt
```

### Local Development
```bash
python webapp/app.py
```

### VS Code Task (Optional)
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Tasks: Run Task"
- Select "Run Diabetes Prediction App"
- Application will start automatically

## 📖 Usage Guide

### Health Parameters Required:
1. **Pregnancies**: Number of pregnancies (0-20)
2. **Glucose**: Blood glucose level in mg/dL (0-300)
3. **Blood Pressure**: Diastolic blood pressure in mmHg (0-200)
4. **Skin Thickness**: Triceps skinfold thickness in mm (0-100)
5. **Insulin**: 2-hour serum insulin in μU/mL (0-1000)
6. **BMI**: Body Mass Index (0-100)
7. **Diabetes Pedigree Function**: Family history factor (0-3)
8. **Age**: Age in years (1-120)

### Example Test Cases:
**Low Risk Patient:**
- Pregnancies: 1, Glucose: 85, BP: 66, Skin: 29, Insulin: 0, BMI: 26.6, DPF: 0.351, Age: 31

**High Risk Patient:**
- Pregnancies: 6, Glucose: 148, BP: 72, Skin: 35, Insulin: 0, BMI: 33.6, DPF: 0.627, Age: 50

## 🔧 Technical Architecture

### Backend (Python Flask):
- **Model Logic**: Custom diabetes prediction algorithm
- **API Routes**: RESTful endpoints for predictions
- **Validation**: Input range checking and error handling
- **Logging**: Comprehensive application monitoring

### Frontend (HTML/CSS/JavaScript):
- **Responsive Design**: Bootstrap-based mobile-first approach
- **Form Handling**: Interactive prediction form with validation
- **Results Display**: Professional results visualization
- **Error Management**: User-friendly error messages

### Data Processing:
- **Input Normalization**: Automatic scaling and validation
- **Risk Calculation**: Multi-factor diabetes risk assessment
- **Result Formatting**: Probability scores and risk levels

## 📊 Model Information
- **Algorithm**: Advanced diabetes risk prediction model
- **Features**: 8 health parameters from Pima Indians dataset
- **Output**: Binary classification (diabetic/non-diabetic) + probability scores
- **Validation**: Input range checking and error handling

## 🚀 Project Status

This project is **ready for deployment** and includes:
1. **Complete Flask Application** - Fully functional web interface
2. **Deployment Configuration** - Ready for cloud platforms
3. **Production Dependencies** - All requirements specified
4. **API Documentation** - Complete endpoint specifications

## 🏆 Project Features
- ✅ **Complete ML Pipeline**
- ✅ **Web Application Interface**
- ✅ **API Implementation**
- ✅ **Data Processing**
- ✅ **Project Documentation**
- ✅ **Responsive Design**
- ✅ **Deployment Ready**

---

*Developed by Hemanth Peddada - July 2025*
