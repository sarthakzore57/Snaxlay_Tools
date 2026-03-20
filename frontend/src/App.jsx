import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

function resolveApiBase() {
  const configuredBase = (import.meta.env.VITE_API_BASE_URL || "").trim();
  if (configuredBase) {
    return configuredBase.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    if (protocol.startsWith("http") && ["localhost", "127.0.0.1"].includes(hostname)) {
      return "http://127.0.0.1:8000";
    }
  }
  return "";
}

const API_BASE = resolveApiBase();
const emptyOrdersState = {
  selected_day: "today",
  available_days: [],
  summary: {
    total_orders: 0,
    total_quantity: 0,
    total_revenue: 0,
    cod_orders: 0,
    prepaid_orders: 0,
  },
  day_buckets: [],
  orders: [],
};

const layoutOptions = [
  { value: 2, title: "2-up stacked", description: "Courier-style format like your sample PDF. Upright labels, top-to-bottom." },
  { value: 4, title: "4-up grid", description: "Fast 2 x 2 print sheet for standard invoice labels." },
  { value: 6, title: "6-up grid", description: "Dense 3 x 2 layout for smaller labels or batch packing." },
];

const detectionOptions = [
  { value: "basic", title: "Basic split", description: "Best for predictable sheets. Fastest mode." },
  { value: "smart", title: "Smart detect", description: "Contour-based detection for mixed layouts." },
];

const paperSizes = ["A4", "A5", "LETTER"];

const stageOrder = ["queued", "starting", "rendering", "rendered", "extracting", "layout", "preview", "completed"];
const workspaceOptions = [
  { value: "print", title: "Print Studio" },
  { value: "orders", title: "Orders Dashboard" },
];

function getErrorMessage(requestError, fallbackMessage) {
  const detail = requestError.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object" && typeof item.msg === "string") {
          return item.msg;
        }
        return "";
      })
      .filter(Boolean);
    if (messages.length) {
      return messages.join(" ");
    }
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return detail.message;
  }
  if (requestError.response?.status) {
    const { status, statusText } = requestError.response;
    return `Request failed (${status}${statusText ? ` ${statusText}` : ""}).`;
  }
  if (typeof requestError.message === "string" && requestError.message.trim()) {
    return requestError.message;
  }
  return fallbackMessage;
}

function estimateProcessingTime(file, detectionMode, layout) {
  if (!file || (Array.isArray(file) && file.length === 0)) {
    return null;
  }
  const files = Array.isArray(file) ? file : [file];
  const totalSize = files.reduce((sum, current) => sum + current.size, 0);
  const sizeInMb = totalSize / (1024 * 1024);
  const modeFactor = detectionMode === "smart" ? 1.7 : 1;
  const layoutFactor = layout === 6 ? 1.12 : layout === 2 ? 0.92 : 1;
  return Math.max(2, Math.round((2.5 + sizeInMb * 2.1) * modeFactor * layoutFactor));
}

function formatCurrency(value) {
  if (typeof value !== "number") {
    return "--";
  }
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatDayLabel(value) {
  if (!value || value === "today") {
    return "Today";
  }
  const parsed = new Date(`${value}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function App() {
  const [activeWorkspace, setActiveWorkspace] = useState("print");
  const [files, setFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [layout, setLayout] = useState(2);
  const [detectionMode, setDetectionMode] = useState("basic");
  const [paperSize, setPaperSize] = useState("A4");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activityLog, setActivityLog] = useState([]);
  const [ordersData, setOrdersData] = useState(emptyOrdersState);
  const [ordersLoading, setOrdersLoading] = useState(false);
  const [ordersError, setOrdersError] = useState("");
  const [selectedOrdersDay, setSelectedOrdersDay] = useState("today");
  const pollRef = useRef(null);

  const estimatedTime = estimateProcessingTime(files, detectionMode, layout);
  const previewPages = result ? result.preview_urls.map((path) => `${API_BASE}${path}`) : [];
  const displayProgress = useMemo(() => {
    if (!job) {
      return uploadProgress;
    }
    return Math.max(uploadProgress, job.progress_percent ?? 0);
  }, [job, uploadProgress]);

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
      }
    };
  }, []);

  useEffect(() => {
    fetchOrders(selectedOrdersDay);
  }, [selectedOrdersDay]);

  const pushLog = (message) => {
    setActivityLog((current) => {
      const next = [`${new Date().toLocaleTimeString()}: ${message}`, ...current];
      return next.slice(0, 8);
    });
  };

  const resetPolling = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const fetchOrders = async (day = "today") => {
    setOrdersLoading(true);
    setOrdersError("");
    try {
      const response = await axios.get(`${API_BASE}/api/orders`, {
        params: { day },
      });
      setOrdersData(response.data);
    } catch (requestError) {
      const fallbackMessage = requestError.code === "ERR_NETWORK"
        ? "Unable to reach the backend API. Check VITE_API_BASE_URL or your deployment routing."
        : "Unable to load the orders dashboard.";
      setOrdersError(getErrorMessage(requestError, fallbackMessage));
    } finally {
      setOrdersLoading(false);
    }
  };

  const handleFileSelection = (selectedFiles) => {
    const incomingFiles = Array.from(selectedFiles || []);
    if (!incomingFiles.length) {
      return;
    }
    for (const selectedFile of incomingFiles) {
      if (selectedFile.type !== "application/pdf" && !selectedFile.name.toLowerCase().endsWith(".pdf")) {
        setError("Only PDF files are supported.");
        return;
      }
    }
    setError("");
    setResult(null);
    setJob(null);
    setActivityLog([]);
    setFiles(incomingFiles);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    handleFileSelection(event.dataTransfer.files);
  };

  const pollJob = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE}/api/jobs/${jobId}`);
      const nextJob = response.data;
      setJob(nextJob);
      if (nextJob.message) {
        setActivityLog((current) => {
          const entry = `${new Date().toLocaleTimeString()}: ${nextJob.message}`;
          if (current[0] === entry) {
            return current;
          }
          return [entry, ...current].slice(0, 8);
        });
      }

      if (nextJob.status === "completed") {
        setResult(nextJob);
        setIsSubmitting(false);
        setUploadProgress(100);
        resetPolling();
        fetchOrders(selectedOrdersDay);
      } else if (nextJob.status === "failed") {
        setError(nextJob.error || "Processing failed.");
        setIsSubmitting(false);
        resetPolling();
      }
    } catch (requestError) {
      const fallbackMessage = requestError.code === "ERR_NETWORK"
        ? "Unable to reach the backend API while checking progress."
        : "Unable to fetch job progress.";
      setError(getErrorMessage(requestError, fallbackMessage));
      setIsSubmitting(false);
      resetPolling();
    }
  };

  const processFile = async () => {
    if (!files.length) {
      setError("Choose at least one PDF file before processing.");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });
    formData.append("layout", String(layout));
    formData.append("detection_mode", detectionMode);
    formData.append("paper_size", paperSize);

    setIsSubmitting(true);
    setUploadProgress(0);
    setError("");
    setResult(null);
    setJob(null);
    setActivityLog([]);
    pushLog("Preparing upload.");

    try {
      const response = await axios.post(`${API_BASE}/api/process`, formData, {
        onUploadProgress: (progressEvent) => {
          if (!progressEvent.total) {
            return;
          }
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        },
      });

      const submittedJob = response.data;
      setUploadProgress(100);
      setJob({
        job_id: submittedJob.job_id,
        status: submittedJob.status,
        stage: "queued",
        message: "Upload complete. Waiting for the worker.",
        progress_percent: 4,
        estimated_processing_seconds: submittedJob.estimated_processing_seconds,
        layout,
        detection_mode: detectionMode,
        paper_size: paperSize,
        warnings: [],
        preview_urls: [],
      });
      pushLog("Upload complete. Background processing started.");

      await pollJob(submittedJob.job_id);
      pollRef.current = window.setInterval(() => {
        pollJob(submittedJob.job_id);
      }, 1200);
    } catch (requestError) {
      const fallbackMessage = requestError.code === "ERR_NETWORK"
        ? "Unable to reach the backend API. Set VITE_API_BASE_URL to your backend or proxy /api to the backend."
        : "Unable to submit the PDF for processing.";
      setError(getErrorMessage(requestError, fallbackMessage));
      setIsSubmitting(false);
    }
  };

  const completedStageIndex = job ? Math.max(stageOrder.indexOf(job.stage), 0) : -1;

  return (
    <div className="app-shell">
      <div className="tech-backdrop" />
      <div className="floating-orb orb-a" />
      <div className="floating-orb orb-b" />
      <div className="floating-orb orb-c" />
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Snaxlay Tools</p>
          <h1>Run label printing and order tracking from one operations dashboard.</h1>
          <p className="hero-text">
            Upload one or more PDFs, extract only the shipping labels, generate print-ready sheets, and review today’s
            order data in a separate dashboard built from the same files.
          </p>
          <div className="hero-actions">
            <div className="hero-chip">Multi-file upload</div>
            <div className="hero-chip">Accurate label crop</div>
            <div className="hero-chip">Day-wise order analytics</div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="signal-card">
            <span>Processing</span>
            <strong>{job?.status === "completed" ? "Ready" : job?.status || "Waiting"}</strong>
          </div>
          <div className="signal-card">
            <span>Orders today</span>
            <strong>{ordersData.summary.total_orders}</strong>
          </div>
          <div className="signal-card">
            <span>Revenue</span>
            <strong>{formatCurrency(ordersData.summary.total_revenue)}</strong>
          </div>
          <div className="signal-card">
            <span>Layout</span>
            <strong>{layout === 2 ? "2-up stacked" : `${layout}-up grid`}</strong>
          </div>
          <div className="signal-card pulse-card">
            <span>Product</span>
            <strong>Snaxlay Tools</strong>
          </div>
        </div>
      </header>

      <div className="workspace-toggle">
        {workspaceOptions.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`workspace-chip ${activeWorkspace === option.value ? "selected" : ""}`}
            onClick={() => setActiveWorkspace(option.value)}
          >
            {option.title}
          </button>
        ))}
      </div>

      {activeWorkspace === "print" ? (
        <main className="dashboard-grid">
        <section className="panel control-panel">
          <div className="panel-header">
            <h2>Upload Labels</h2>
            <span className="status-pill">{isSubmitting ? "Processing" : "Ready"}</span>
          </div>

          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              id="pdf-input"
              type="file"
              multiple
              accept="application/pdf,.pdf"
              onChange={(event) => handleFileSelection(event.target.files)}
            />
            <label htmlFor="pdf-input">
              <span className="dropzone-title">
                {files.length ? `${files.length} PDF file(s) selected` : "Drag shipping-label PDFs into the pipeline"}
              </span>
              <span className="dropzone-subtitle">
                PDF only. Multi-page invoices and batch uploads are supported.
              </span>
            </label>
          </div>
          {files.length ? (
            <div className="file-list">
              {files.slice(0, 6).map((file) => (
                <p key={file.name + file.size}>{file.name}</p>
              ))}
              {files.length > 6 ? <p>+ {files.length - 6} more file(s)</p> : null}
            </div>
          ) : null}

          <div className="control-group">
            <h3>Layout Matrix</h3>
            <div className="option-grid">
              {layoutOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`option-card ${layout === option.value ? "selected" : ""}`}
                  onClick={() => setLayout(option.value)}
                >
                  <strong>{option.title}</strong>
                  <span>{option.description}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="control-group">
            <h3>Detection Engine</h3>
            <div className="option-grid">
              {detectionOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`option-card ${detectionMode === option.value ? "selected" : ""}`}
                  onClick={() => setDetectionMode(option.value)}
                >
                  <strong>{option.title}</strong>
                  <span>{option.description}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="control-group">
            <h3>Output Sheet</h3>
            <div className="chip-row">
              {paperSizes.map((value) => (
                <button
                  key={value}
                  type="button"
                  className={`chip ${paperSize === value ? "selected" : ""}`}
                  onClick={() => setPaperSize(value)}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>

          {estimatedTime ? <div className="estimate-banner">Estimated processing time: about {estimatedTime} seconds.</div> : null}
          {error ? <div className="error-banner">{error}</div> : null}

          <button type="button" className="primary-button" disabled={isSubmitting} onClick={processFile}>
            {isSubmitting ? "Background Processing Active" : "Launch Processing Job"}
          </button>
        </section>

        <section className="panel telemetry-panel">
          <div className="panel-header">
            <h2>Processing Status</h2>
            <span className={`status-pill ${job?.status === "completed" ? "success" : ""}`}>
              {job?.stage || "idle"}
            </span>
          </div>

          <div className="telemetry-card">
            <div className="progress-meta">
              <div>
                <span className="metric-label">Current status</span>
                <strong>{job?.message || "Waiting for files."}</strong>
              </div>
              <div className="metric-number">{displayProgress}%</div>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${displayProgress}%` }} />
            </div>
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-label">Estimated</span>
                <strong>{job?.estimated_processing_seconds ? `${Math.round(job.estimated_processing_seconds)}s` : "--"}</strong>
              </div>
              <div className="metric-card">
                <span className="metric-label">Actual</span>
                <strong>{job?.actual_processing_seconds ? `${job.actual_processing_seconds}s` : "--"}</strong>
              </div>
              <div className="metric-card">
                <span className="metric-label">Pages</span>
                <strong>{job?.source_page_count ?? "--"}</strong>
              </div>
              <div className="metric-card">
                <span className="metric-label">Labels</span>
                <strong>{job?.extracted_label_count ?? "--"}</strong>
              </div>
            </div>
          </div>

          <div className="stage-rail">
            {stageOrder.map((stage, index) => (
              <div
                key={stage}
                className={`stage-node ${index <= completedStageIndex ? "active" : ""} ${job?.stage === stage ? "current" : ""}`}
              >
                <span className="stage-dot" />
                <span className="stage-label">{stage}</span>
              </div>
            ))}
          </div>

          <div className="activity-feed">
            <div className="feed-header">
              <h3>Recent Updates</h3>
              <span>{activityLog.length ? "active" : "standby"}</span>
            </div>
            {activityLog.length ? (
              <div className="feed-list">
                {activityLog.map((entry) => (
                  <p key={entry}>{entry}</p>
                ))}
              </div>
            ) : (
              <div className="feed-empty">Processing updates will appear here after you start.</div>
            )}
          </div>
        </section>

        <section className="panel preview-panel">
          <div className="panel-header">
            <div>
              <h2>Output Preview</h2>
              <p className="panel-subtitle">Review the generated print sheets before download.</p>
            </div>
            {result?.download_url ? (
              <a className="primary-button compact" href={`${API_BASE}${result.download_url}`}>
                Download PDF
              </a>
            ) : null}
          </div>

          {result ? (
            <>
              <div className="summary-grid">
                <div className="summary-card">
                  <span>Output pages</span>
                  <strong>{result.output_page_count}</strong>
                </div>
                <div className="summary-card">
                  <span>Detection mode</span>
                  <strong>{result.detection_mode}</strong>
                </div>
                <div className="summary-card">
                  <span>Paper</span>
                  <strong>{result.paper_size}</strong>
                </div>
              </div>

              {result.warnings?.length ? (
                <div className="warning-list">
                  {result.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}

              <div className="preview-pages">
                {previewPages.map((src, index) => (
                  <figure key={src} className="preview-sheet">
                    <img src={src} alt={`Preview page ${index + 1}`} />
                    <figcaption>Sheet {index + 1}</figcaption>
                  </figure>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-preview">
              <p>No output rendered yet.</p>
              <span>The preview deck appears here as soon as the worker completes the print layout.</span>
            </div>
          )}
        </section>
        </main>
      ) : (
        <main className="orders-grid">
          <section className="panel orders-summary-panel">
            <div className="panel-header">
              <div>
                <h2>Orders Dashboard</h2>
                <p className="panel-subtitle">Track day-wise order volume, value, and label-ready inventory from uploaded PDFs.</p>
              </div>
              <button type="button" className="primary-button compact" onClick={() => fetchOrders(selectedOrdersDay)}>
                Refresh Data
              </button>
            </div>

            <div className="day-filter-row">
              <button
                type="button"
                className={`chip ${selectedOrdersDay === "today" ? "selected" : ""}`}
                onClick={() => setSelectedOrdersDay("today")}
              >
                Today
              </button>
              {ordersData.available_days.map((day) => (
                <button
                  key={day}
                  type="button"
                  className={`chip ${selectedOrdersDay === day ? "selected" : ""}`}
                  onClick={() => setSelectedOrdersDay(day)}
                >
                  {formatDayLabel(day)}
                </button>
              ))}
            </div>

            {ordersError ? <div className="error-banner">{ordersError}</div> : null}

            <div className="orders-summary-grid">
              <div className="summary-card emphasis">
                <span>Selected Day</span>
                <strong>{formatDayLabel(ordersData.selected_day)}</strong>
              </div>
              <div className="summary-card">
                <span>Total Orders</span>
                <strong>{ordersData.summary.total_orders}</strong>
              </div>
              <div className="summary-card">
                <span>Total Quantity</span>
                <strong>{ordersData.summary.total_quantity}</strong>
              </div>
              <div className="summary-card">
                <span>Total Revenue</span>
                <strong>{formatCurrency(ordersData.summary.total_revenue)}</strong>
              </div>
              <div className="summary-card">
                <span>COD Orders</span>
                <strong>{ordersData.summary.cod_orders}</strong>
              </div>
              <div className="summary-card">
                <span>Prepaid Orders</span>
                <strong>{ordersData.summary.prepaid_orders}</strong>
              </div>
            </div>

            <div className="day-buckets-grid">
              {ordersData.day_buckets.length ? (
                ordersData.day_buckets.map((bucket) => (
                  <button
                    key={bucket.day}
                    type="button"
                    className={`day-bucket ${selectedOrdersDay === bucket.day ? "selected" : ""}`}
                    onClick={() => setSelectedOrdersDay(bucket.day)}
                  >
                    <span>{formatDayLabel(bucket.day)}</span>
                    <strong>{bucket.order_count} orders</strong>
                    <small>{formatCurrency(bucket.revenue)}</small>
                  </button>
                ))
              ) : (
                <div className="empty-preview orders-empty-card">
                  <p>No orders extracted yet.</p>
                  <span>Process a PDF in Print Studio and this dashboard will populate automatically.</span>
                </div>
              )}
            </div>
          </section>

          <section className="panel orders-table-panel">
            <div className="panel-header">
              <div>
                <h2>Order Lines</h2>
                <p className="panel-subtitle">Structured fields extracted directly from your shipping invoices and labels.</p>
              </div>
              <span className="status-pill">{ordersLoading ? "Loading" : `${ordersData.orders.length} records`}</span>
            </div>

            {ordersLoading ? (
              <div className="empty-preview">
                <p>Refreshing dashboard data.</p>
                <span>The latest extracted orders are being loaded.</span>
              </div>
            ) : ordersData.orders.length ? (
              <>
                <div className="orders-table-wrapper">
                  <table className="orders-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Vendor</th>
                        <th>Customer</th>
                        <th>Product</th>
                        <th>Qty</th>
                        <th>Price</th>
                        <th>Delivery</th>
                        <th>Order ID</th>
                        <th>Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ordersData.orders.map((order) => (
                        <tr key={order.order_key}>
                          <td>{formatDayLabel(order.order_day)}</td>
                          <td>{order.vendor}</td>
                          <td>
                            <strong>{order.customer_name || "--"}</strong>
                            <span>{[order.city, order.state].filter(Boolean).join(", ") || order.platform || "--"}</span>
                          </td>
                          <td>
                            <strong>{order.product_name || "--"}</strong>
                            <span>{order.sku_code || order.suborder_id || "--"}</span>
                          </td>
                          <td>{order.quantity ?? "--"}</td>
                          <td>{formatCurrency(order.price)}</td>
                          <td>{order.delivery_option || "--"}</td>
                          <td>{order.order_id || order.invoice_number || "--"}</td>
                          <td>
                            <strong>{order.source_file}</strong>
                            <span>Page {order.source_page}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="orders-cards">
                  {ordersData.orders.map((order) => (
                    <article key={order.order_key} className="order-card">
                      <div className="order-card-top">
                        <span className="order-card-tag">{order.vendor}</span>
                        <span className="order-card-day">{formatDayLabel(order.order_day)}</span>
                      </div>
                      <h3>{order.product_name || "Unlabeled product"}</h3>
                      <p>{order.customer_name || "Unknown customer"}</p>
                      <div className="order-card-meta">
                        <span>Qty {order.quantity ?? "--"}</span>
                        <span>{formatCurrency(order.price)}</span>
                        <span>{order.delivery_option || "--"}</span>
                      </div>
                      <div className="order-card-foot">
                        <span>{order.order_id || order.invoice_number || "--"}</span>
                        <span>{order.source_file} • Page {order.source_page}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-preview">
                <p>No orders available for {formatDayLabel(ordersData.selected_day)}.</p>
                <span>Switch the day filter or process more PDFs to build the dashboard.</span>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
