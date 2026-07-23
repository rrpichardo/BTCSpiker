import { usePolling } from "../usePolling.js";
import { fetchSystemStatus } from "../api.js";
import StatusCard from "../components/StatusCard.jsx";
import { formatAge } from "../format.js";

function overallStatus(services, stale) {
  if (stale) return { label: "Status stale", className: "pill-amber" };
  if (services.length === 0) return { label: "No services", className: "pill-gray" };
  if (services.some((service) => !service.ok)) {
    return { label: "Service down", className: "pill-red" };
  }
  if (services.some((service) => service.degraded)) {
    return { label: "Degraded", className: "pill-amber" };
  }
  return { label: "Operational", className: "pill-green" };
}

export default function SystemPage() {
  const { data, error, lastUpdated, isLoading, isRefreshing } = usePolling(
    fetchSystemStatus,
    5000,
  );
  const services = data?.services ?? [];
  const hasResponse = data !== null;
  const cached = Boolean(error && hasResponse);
  const overall = overallStatus(services, cached);

  return (
    <section className="page" id="system-page" aria-labelledby="system-heading">
      <div className="page-header page-header-split">
        <div>
          <p className="eyebrow">Pipeline control plane</p>
          <h2 id="system-heading">System status</h2>
          <p className="page-header-note">API, observability, model registry & prediction log</p>
        </div>
        <div className="system-summary" aria-live="polite">
          {hasResponse && <span className={`pill ${overall.className}`}>{overall.label}</span>}
          <span className="page-header-note">
            {formatAge(lastUpdated)}{isRefreshing ? " · checking" : ""}
          </span>
        </div>
      </div>

      {isLoading && !hasResponse && (
        <p className="state-message" role="status" aria-live="polite">Checking pipeline services…</p>
      )}
      {error && (
        <p className="stale-banner" role="alert">
          {hasResponse
            ? "The status refresh failed. Service cards are cached and have been marked stale."
            : "System status is unavailable. Confirm the API is reachable and retry."}
        </p>
      )}

      <div className="status-grid">
        {services.map((s) => (
          <StatusCard key={s.name} service={s} stale={cached} />
        ))}
        {hasResponse && services.length === 0 && !error && (
          <p className="state-message">Connected successfully; no services were reported.</p>
        )}
      </div>
    </section>
  );
}
