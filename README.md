# Spam Text Detector

A web application that uses machine learning to detect spam texts and analyze URLs for safety.

## Features

- Text analysis to detect spam messages
- URL extraction and safety analysis
- Interactive visualizations with Chart.js
- Mobile-responsive design

## Project Structure

```
spam-text-detector/
│
├── main.py                  # FastAPI application
├── requirements.txt         # Python dependencies
├── text_classification.pkl  # ML model file (needs to be provided)
│
├── templates/               # HTML templates
│   └── index.html           # Main application page
│
└── static/                  # Static files
    ├── css/
    │   └── styles.css       # CSS styles
    └── js/
        └── main.js          # JavaScript for frontend functionality
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository or download the source code

2. Create and activate a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Place your trained ML model file (`text_classification.pkl`) in the root directory

### Running the Application

1. Start the FastAPI server:

   ```
   uvicorn main:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Docker Deployment

You can also run the application using Docker:

1. Build the Docker image:

   ```
   docker build -t spam-detector .
   ```

2. Run the container:

   ```
   docker run -p 8000:8000 spam-detector
   ```

3. Access the application at `http://localhost:8000`

## API Endpoints

- `GET /`: Main web interface
- `POST /api/analyze`: Analyze text for spam and URL safety
  - Request body: `{"text": "text to analyze"}`
- `GET /api/demo-text`: Get a sample demonstration text

## Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js
- **ML**: Scikit-learn (for the model)

## License

This project is licensed under the MIT License.
