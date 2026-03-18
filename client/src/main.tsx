import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return React.createElement(
        "pre",
        { style: { padding: "2rem", color: "red", whiteSpace: "pre-wrap" } },
        `Runtime error:\n${this.state.error.message}\n\n${this.state.error.stack}`
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
