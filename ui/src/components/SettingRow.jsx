import { useState } from "react";

export default function SettingRow({ setting }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(setting.editable_via).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div className="setting-row">
      <div className="setting-row-header">
        <span className="setting-key">{setting.key}</span>
        <div className="setting-badges">
          {setting.apply_state === "restart_required" && (
            <span className="pill pill-amber">restart required</span>
          )}
          {setting.apply_state === "unknown" && (
            <span className="pill pill-gray">unknown</span>
          )}
          {setting.danger && <span className="pill pill-red">requires retraining</span>}
        </div>
      </div>

      {setting.description && <p className="setting-description">{setting.description}</p>}

      <div className="setting-values">
        <div className="setting-value">
          <span className="setting-value-label">saved</span>
          <code>{String(setting.saved_value)}</code>
        </div>
        <div className="setting-value">
          <span className="setting-value-label">active</span>
          <code>{String(setting.active_value)}</code>
        </div>
        <div className="setting-value">
          <span className="setting-value-label">source</span>
          <code>{setting.source}</code>
        </div>
      </div>

      {setting.note && <p className="setting-note">{setting.note}</p>}

      <div className="setting-editable-via">
        <code>{setting.editable_via}</code>
        <button type="button" className="copy-btn" onClick={handleCopy}>
          {copied ? "copied" : "copy"}
        </button>
      </div>
    </div>
  );
}
