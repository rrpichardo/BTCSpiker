// Accessible "i" info affordance. The full text lives in aria-label so
// screen readers get it on focus without needing the visual tooltip; the
// tooltip span is aria-hidden since it would otherwise be announced twice.
export default function InfoIcon({ label }) {
  return (
    <span className="info-icon-wrap">
      <button type="button" className="info-icon" aria-label={label}>
        <span aria-hidden="true">i</span>
      </button>
      <span className="info-icon-tooltip" aria-hidden="true">
        {label}
      </span>
    </span>
  );
}
