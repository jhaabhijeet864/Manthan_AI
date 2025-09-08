# Backend Server

This directory contains the Node.js and Express backend for the Ocean Intelligence Platform.

## Overview

The backend is a simple Express server responsible for serving predictions from the Python machine learning models. It provides a RESTful API that the frontend can consume.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v14 or later)
- [npm](https://www.npmjs.com/)

### Installation

1.  **Navigate to the server directory:**
    ```bash
    cd server
    ```

2.  **Install dependencies:**
    This will install all the necessary packages from `package.json`.
    ```bash
    npm install
    ```

## Available Scripts

### `npm run dev`

Starts the server in development mode using `nodemon`.
The server will automatically restart if you make any changes to the code.
The API will be available at `http://localhost:5000`.

### `npm start`

Starts the server in production mode.

## API Documentation

The server exposes the following endpoints for the application.

---

### Health Check

- **Endpoint:** `/api/health`
- **Method:** `GET`
- **Description:** A simple health check endpoint to verify that the server is running.
- **Response:**
  - **200 OK**
    ```json
    {
      "status": "ok",
      "service": "CMLRE Marine Platform Backend"
    }
    ```

---

### Get Species Prediction

- **Endpoint:** `/api/predict`
- **Method:** `GET`
- **Description:** Triggers the execution of the species distribution predictive model (`predictive_model.py`). This script runs the machine learning model and returns the prediction results.
- **Request Body:** None
- **Response:**
  - **200 OK**
    A JSON object containing the prediction results. The format is determined by the output of the Python script.
    
    *Example Response:*
    ```json
    {
      "prediction": [
        {
          "species": "Species A",
          "probability": 0.85,
          "coordinates": {
            "lat": -45.123,
            "lon": 170.456
          }
        },
        {
          "species": "Species B",
          "probability": 0.72,
          "coordinates": {
            "lat": -46.789,
            "lon": 168.123
          }
        }
      ],
      "model_info": {
        "model_name": "Random Forest Classifier",
        "timestamp": "2023-10-27T10:00:00Z"
      }
    }
    ```
  - **500 Internal Server Error**
    If the Python script fails to execute or returns an error.
    
    *Example Error Response:*
    ```json
    {
      "error": "Prediction failed",
      "details": "Error message from the Python script."
    }
    ```

## Project Structure

- **/src**: Contains the main application logic.
  - **/controllers**: Handles the request/response logic for API routes.
  - **/routes**: Defines the API endpoints.
  - **/services**: Contains business logic, such as interacting with the ML scripts.
- **app.js**: The main Express application file where routes and middleware are configured.
- **server.js**: The entry point for starting the server.
