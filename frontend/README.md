# Sign Language App Frontend

This is the Next.js frontend for the Sign Language Learning application. It connects to the FastAPI backend via WebSockets to provide real-time gesture feedback.

## Architecture

- **Framework**: [Next.js 16](https://nextjs.org) (App Router)
- **Styling**: Tailwind CSS v4
- **State Management**: React Hooks (useState, useEffect, useRef)
- **Communication**: Native WebSockets (`src/lib/hooks/useWebSocket.ts`)

The frontend is designed to be fully static-exportable (`output: 'export'`), meaning it can be served directly by the FastAPI backend in production, eliminating the need for a separate Node.js server.

## Getting Started

First, ensure your FastAPI backend is running (which handles the WebSocket connection and model inference).

Then, in this directory, run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Development Guide

### Key Components

- `app/page.tsx`: The main landing page and application entry point.
- `components/WebcamView.tsx`: Handles webcam capture, drawing hand landmarks via canvas, and passing frames to the WebSocket.
- `components/Dashboard.tsx`: Displays session statistics, accuracy, and attempt counts.
- `lib/api.ts`: API helper functions for fetching tutorial assets and managing sessions.

### WebSocket Integration

The app uses a custom hook `useWebSocket` that maintains a persistent connection to `ws://localhost:8000/ws`.
- It sends base64 encoded frames during the "recording" phase.
- It receives prediction results, confidence scores, and session state updates.

## Production Build

To build the frontend for production:

```bash
npm run build
```

This will generate an `out/` directory containing the static HTML/CSS/JS files. The backend (`main.py`) will automatically serve these files.
