import { useState } from "react";
import PredictionsPage from "./pages/PredictionsPage.jsx";
import SettingsPage from "./pages/SettingsPage.jsx";
import SystemPage from "./pages/SystemPage.jsx";

const TABS = [
  { id: "predictions", label: "Predictions" },
  { id: "settings", label: "Settings" },
  { id: "system", label: "System" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predictions");

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-name">BTCSpiker</span>
        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab-btn ${activeTab === tab.id ? "tab-btn-active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {/* Only the active page mounts, so hidden tabs stop polling entirely. */}
        {activeTab === "predictions" && <PredictionsPage />}
        {activeTab === "settings" && <SettingsPage />}
        {activeTab === "system" && <SystemPage />}
      </main>
    </div>
  );
}
