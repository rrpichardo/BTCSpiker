import { usePolling } from "../usePolling.js";
import { fetchSettings } from "../api.js";
import SettingRow from "../components/SettingRow.jsx";

export default function SettingsPage() {
  const { data, error } = usePolling(fetchSettings, 10000);
  const settings = data?.settings ?? [];

  return (
    <div className="page">
      <div className="page-header">
        <h2>Settings</h2>
        <p className="page-header-note">
          Settings are read-only here. To change a value, run the command
          shown for that setting on the host, then restart the affected
          service if indicated.
        </p>
      </div>

      {error && <p className="error-text">Failed to load settings: {error.message}</p>}

      <div className="settings-list">
        {settings.map((s) => (
          <SettingRow key={s.key} setting={s} />
        ))}
        {settings.length === 0 && !error && <p className="empty-cell">No settings found.</p>}
      </div>
    </div>
  );
}
