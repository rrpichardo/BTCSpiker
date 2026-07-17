import { usePolling } from "../usePolling.js";
import { fetchSystemStatus } from "../api.js";
import StatusCard from "../components/StatusCard.jsx";

export default function SystemPage() {
  const { data, error, lastUpdated } = usePolling(fetchSystemStatus, 5000);
  const services = data?.services ?? [];

  return (
    <div className="page">
      <div className="page-header">
        <h2>System status</h2>
        {lastUpdated && (
          <p className="page-header-note">Checked at {lastUpdated.toLocaleTimeString()}</p>
        )}
      </div>

      {error && <p className="error-text">Failed to load system status: {error.message}</p>}

      <div className="status-grid">
        {services.map((s) => (
          <StatusCard key={s.name} service={s} />
        ))}
        {services.length === 0 && !error && <p className="empty-cell">No services reported.</p>}
      </div>
    </div>
  );
}
