function statusInfo(service) {
  if (!service.ok) return { label: "down", className: "pill-red" };
  if (service.degraded) return { label: "degraded", className: "pill-amber" };
  return { label: "ok", className: "pill-green" };
}

export default function StatusCard({ service }) {
  const status = statusInfo(service);

  return (
    <div className="status-card">
      <div className="status-card-header">
        <span className="status-card-name">{service.name}</span>
        <span className={`pill ${status.className}`}>{status.label}</span>
      </div>
      <div className="status-card-body">
        <div className="status-card-row">
          <span className="status-card-label">latency</span>
          <span>{typeof service.latency_ms === "number" ? `${service.latency_ms.toFixed(1)} ms` : "-"}</span>
        </div>
        {service.detail && <p className="status-card-detail">{service.detail}</p>}
      </div>
      {service.open_url && (
        <a
          className="status-card-link"
          href={service.open_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open &rarr;
        </a>
      )}
    </div>
  );
}
