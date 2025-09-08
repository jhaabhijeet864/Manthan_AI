# Frontend Application

This directory contains the source code for the React-based frontend of the Ocean Intelligence Platform.

## Overview

The frontend is a single-page application (SPA) built with [React](https://reactjs.org/). It provides an interactive interface for visualizing oceanographic data and the predictions from the machine learning models.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v14 or later)
- [npm](https://www.npmjs.org/)

### Installation

1.  **Navigate to the client directory:**
    ```bash
    cd client
    ```

2.  **Install dependencies:**
    This command will download and install all the necessary packages defined in `package.json`.
    ```bash
    npm install
    ```

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in development mode.
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

## Project Structure

- **/public**: Contains the main `index.html` file and other static assets.
- **/src**: Contains the main React application source code.
  - **/components**: Reusable UI components (Header, Navigation, etc.).
  - **/hooks**: Custom React hooks, such as for chart rendering.
  - **/services**: Modules for making API calls to the backend.
  - **/App.js**: The main application component.
  - **/index.js**: The entry point of the application.

