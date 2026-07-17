import { useEffect, useId, useRef, useState } from "react";

function copyWithFallback(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  let copied = false;
  try {
    textarea.select();
    copied = document.execCommand("copy");
  } finally {
    textarea.remove();
  }
  if (!copied) throw new Error("Browser copy command was rejected");
}

export default function SettingRow({ setting }) {
  const [copyState, setCopyState] = useState("idle");
  const feedbackId = useId();
  const timerRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  function resetCopyStateLater() {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (mountedRef.current) setCopyState("idle");
    }, 2500);
  }

  async function handleCopy() {
    try {
      if (window.isSecureContext && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(setting.editable_via);
      } else {
        copyWithFallback(setting.editable_via);
      }
      if (!mountedRef.current) return;
      setCopyState("copied");
    } catch {
      if (!mountedRef.current) return;
      setCopyState("error");
    }
    resetCopyStateLater();
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
        <input
          type="text"
          name={`${setting.key}-apply-command`}
          value={setting.editable_via}
          readOnly
          autoComplete="off"
          spellCheck="false"
          aria-label={`Apply command for ${setting.key}`}
          onFocus={(event) => event.currentTarget.select()}
        />
        <button
          type="button"
          className="copy-btn"
          onClick={handleCopy}
          aria-describedby={feedbackId}
        >
          {copyState === "copied" ? "Copied" : "Copy command"}
        </button>
      </div>
      <span
        id={feedbackId}
        className={copyState === "error" ? "copy-feedback copy-feedback-error" : "copy-feedback"}
        role="status"
        aria-live="polite"
      >
        {copyState === "copied" && "Command copied to clipboard."}
        {copyState === "error" && "Copy failed. Select the command and copy it manually."}
      </span>
    </div>
  );
}
