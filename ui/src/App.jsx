import { useState } from "react";
import PredictionsPage from "./pages/PredictionsPage.jsx";
import SettingsPage from "./pages/SettingsPage.jsx";
import SystemPage from "./pages/SystemPage.jsx";
import PerformancePage from "./pages/PerformancePage.jsx";
import TournamentPage from "./pages/TournamentPage.jsx";

const TABS = [
  { id: "predictions", label: "Predictions" },
  { id: "settings", label: "Settings" },
  { id: "system", label: "System" },
  { id: "performance", label: "Performance" },
  { id: "tournament", label: "Tournament" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predictions");

  return (
    <div className="app">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <header className="app-header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">
            ₿
          </span>
          <div>
            <h1 className="app-name">BTCSpiker</h1>
            <p className="app-tagline">Volatility signal operations</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Primary">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab-btn ${activeTab === tab.id ? "tab-btn-active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              aria-current={activeTab === tab.id ? "page" : undefined}
              aria-pressed={activeTab === tab.id}
              aria-controls={activeTab === tab.id ? `${tab.id}-page` : undefined}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main" id="main-content" tabIndex="-1">
        {/* Only the active page mounts, so hidden tabs stop polling entirely. */}
        {activeTab === "predictions" && <PredictionsPage />}
        {activeTab === "settings" && <SettingsPage />}
        {activeTab === "system" && <SystemPage />}
        {activeTab === "performance" && <PerformancePage />}
        {activeTab === "tournament" && <TournamentPage />}
      </main>
    </div>
  );
}
