
  import { createRoot } from "react-dom/client";
  import App from "./app/App.tsx";
  import { startBootDeadline } from "./app/lib/boot";
  import "./styles/index.css";

  // If a step never reports, the app is shown anyway rather than leaving
  // someone watching a progress bar with nothing behind it.
  startBootDeadline();

  createRoot(document.getElementById("root")!).render(<App />);
  