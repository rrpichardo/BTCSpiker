import { usePolling } from "../usePolling.js";
import { fetchSettings } from "../api.js";
import SettingRow from "../components/SettingRow.jsx";
import { formatAge } from "../format.js";

export default function SettingsPage() {
  const { data, error, lastUpdated, isLoading, isRefreshing } = usePolling(
    fetchSettings,
    10000,
  );
  const settings = data?.settings ?? [];
  const hasResponse = data !== null;

  return (
    <section className="page" id="settings-page" aria-labelledby="settings-heading">
      <div className="page-header page-header-split">
        <div>
          <p className="eyebrow">Read-only control map</p>
          <h2 id="settings-heading">Settings</h2>
        </div>
        <p className="page-header-note page-header-freshness">
          {formatAge(lastUpdated)}{isRefreshing ? " · refreshing" : ""}
        </p>
      </div>
      <div className="settings-intro">
        <p className="page-header-note">
          Compare values saved on disk with the configuration active in the running
          pipeline. Apply changes on the host using the command shown for each setting.
        </p>
      </div>

      {isLoading && !hasResponse && (
        <p className="state-message" role="status" aria-live="polite">Loading settings…</p>
      )}
      {error && (
        <p className="stale-banner" role="alert">
          {hasResponse
            ? `Settings refresh failed. Showing cached values; ${formatAge(lastUpdated).toLowerCase()}.`
            : "Settings are unavailable. Check the API service and retry."}
        </p>
      )}

      <div className="settings-list">
        {settings.map((s) => (
          <SettingRow key={s.key} setting={s} />
        ))}
        {hasResponse && settings.length === 0 && !error && (
          <p className="state-message">Connected successfully; no settings were reported.</p>
        )}
      </div>
    </section>
  );
}
