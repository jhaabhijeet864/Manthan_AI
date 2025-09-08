# Ocean Intelligence Platform

This project is a full-stack application designed for oceanographic data analysis and species distribution prediction. It features a React-based frontend for data visualization, a Node.js/Express backend to serve predictions, and a collection of Python scripts for machine learning model training and data processing.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Frontend](#running-the-frontend)
  - [Running the Backend](#running-the-backend)
  - [Machine Learning Scripts](#machine-learning-scripts)
- [API Documentation](#api-documentation)
- [Data](#data)
- [Contributing](#contributing)

## Project Structure

The repository is organized into the following main directories:

- **/client**: Contains the React frontend application for data visualization and user interaction.
- **/server**: The Node.js and Express backend server that exposes the API for ML model predictions.
- **/models**: Includes Python scripts for data fusion, processing, and training the machine learning models.
- **/scripts**: Utility scripts for various tasks, including interacting with the trained models.
- **/data**: Holds raw and processed data used for model training and application display.

## Getting Started

### Prerequisites

Make sure you have the following installed on your system:

- [Node.js](https://nodejs.org/) (v14 or later)
- [npm](https://www.npmjs.com/) (comes with Node.js)
- [Python](https://www.python.org/) (v3.8 or later)
- [pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Frontend Dependencies:**
    ```bash
    cd client
    npm install
    cd ..
    ```

3.  **Install Backend Dependencies:**
    ```bash
    cd server
    npm install
    cd ..
    ```

4.  **Install Python Dependencies:**
    Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install the required packages from the `models` and root directories:
    ```bash
    pip install -r requirements.txt
    pip install -r models/requirements.txt
    ```

## AI Integration Roadmap (2025)

### Step 1: Video Frame Extraction
Extract frames from all .mp4 videos using OpenCV. Run:
```bash
python models/extract_frames.py
```
Frames are saved in `models/data/raw_data/images/positive` and `models/data/raw_data/images/negative`.

### Step 2: Dataset Organization
Split extracted frames into train/val folders for CNN training:
```bash
python models/split_images.py
```
This creates:
```
models/data/raw_data/images/
  train/
    positive/
    negative/
  val/
    positive/
    negative/
```

### Step 3: Data Preprocessing and Fusion
Merge CMLRE and WOD datasets for tabular ML using:
```bash
python models/data_fusion.py
```
Output is saved in `data/processed_data/`.

### Step 4: Model Building and Training
Train your CNN on the organized image dataset:
```bash
python models/train_cnn.py
```
This saves the model in both HDF5 and Keras formats:
```
models/trained_models/fish_cnn.h5
models/trained_models/fish_cnn.keras
```

Train tabular models using scripts in `scripts/utils/` as needed.

### Step 5: API Deployment and Frontend Integration
Wrap trained models in a Flask or FastAPI API. The Node.js backend communicates with these Python microservices, and the React frontend interacts with the Node.js backend.

---

## Usage

### Running the Frontend

To start the React development server:

```bash
cd client
npm start
```
The application will be available at `http://localhost:3000`.

### Running the Backend

To start the Node.js backend server:

```bash
cd server
npm run dev
```
The API server will start on `http://localhost:5000`.


### Machine Learning Scripts

See the AI Integration Roadmap above for the recommended workflow. For more details, see [models/README.md](models/README.md).

## API Documentation

The backend provides a simple API for predictions. For detailed information, please see the [server/README.md](server/README.md) file, which contains the full API documentation.

## Data

The `data` directory contains the datasets used in this project.
- `data/raw_data`: Original, unmodified data files.
- `data/processed_data`: Cleaned and fused data ready for model training.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.