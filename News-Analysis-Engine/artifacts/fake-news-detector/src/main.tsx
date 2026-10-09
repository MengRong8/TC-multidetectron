import { createRoot } from "react-dom/client";
import { setBaseUrl } from "@workspace/api-client-react";
import App from "./App";
import "./index.css";

// Always use same-origin API path. Vite dev server proxies /api to backend gateway.
// This avoids localhost mismatches in port-forwarded or remote preview environments.
const API_BASE_URL = "";
setBaseUrl(API_BASE_URL);
console.log("[API Client] Configured to use: (same-origin /api)");

createRoot(document.getElementById("root")!).render(<App />);
