# Desktop startup progress protocol

Before the local HTTP server is available, the QuantEM sidecar reports startup
liveness over standard error. This keeps the established, script-readable
server URL and data-directory banner on standard output unchanged. Each event is one line beginning with
`QUANTEM_STARTUP_PROGRESS ` followed by a JSON object. Other output remains
human-readable and may be logged normally.

Every phase uses the same event shape:

| Field | Meaning |
|---|---|
| `phase` | Stable machine-readable phase name. |
| `category` | `startup`, `database-upgrade`, `database-recovery`, `ready`, or `failure`. |
| `message` | Short primary status for the startup screen. |
| `detail` | Optional user-facing explanation of the current work. |
| `critical` / `can_close` | Whether closing can interrupt database work. |
| `completed` / `total` / `unit` | Measured work, when a real denominator exists. |
| `percent` | Derived percentage when the operation is determinate; otherwise `null`. |
| `elapsed_seconds` | Time since this startup began. |
| `phase_elapsed_seconds` | Time spent in the current phase. |
| `last_progress_seconds` | Time since measurable progress, distinct from liveness. |
| `heartbeat` | `true` when the event confirms liveness without claiming progress. |
| `technical_detail` | Optional diagnostic detail hidden behind disclosure UI. |

Emitters must flush each event immediately and continue sending heartbeats
during opaque or long-running calls. Heartbeats must not advance a progress
bar. Consumers must ignore unknown fields and phases so the protocol can grow
without coupling a desktop-shell release to every sidecar change.

Database work has no total wall-clock deadline after the protocol establishes
liveness. A missing initial event may still time out. When measurable progress
stops, the UI may surface a *stalled* warning, but a live database process is
left running; an actual process exit or I/O error is what fails startup.
Failure events use `phase: "failed"`, name the `failed_phase`, and make the
startup screen safe to close.
