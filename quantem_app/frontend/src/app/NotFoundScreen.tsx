/**
 * What the app says when it is handed an address it does not know.
 *
 * The route table used to end in `<Navigate to="/" replace />`: a nonsense
 * address silently became the library. That is recoverable but dishonest --
 * nothing tells the reader their link was wrong, so a mistyped or stale
 * bookmark looks like the app losing their place, and a link shared between two
 * builds fails without saying it failed.
 *
 * Two rules this screen keeps:
 *
 * * **It never echoes the address.** The obvious version of this page prints
 *   what you typed, and on this app that address is usually
 *   `#/assets/<uuid>/…`. A raw identifier is not something a person can act on
 *   and it is exactly what the copy invariant forbids in user-facing text, so
 *   the page describes the situation instead of quoting it.
 * * **It says the work is safe.** Landing somewhere blank while a segmentation
 *   is running is frightening in a way the actual fault does not deserve:
 *   routing has no bearing on stored objects, and the sentence that says so
 *   costs one line.
 */

export function NotFoundScreen() {
  return (
    <div className="not-found" role="main" data-testid="not-found-screen">
      <h1 className="not-found-title">This page does not exist</h1>
      <p className="not-found-body">
        The address you opened is not one QuantEM knows. It was probably typed
        by hand, or saved from an older version of the app.
      </p>
      <p className="not-found-body">
        Nothing has happened to your images or your objects. They are in the
        library, exactly as you left them.
      </p>
      <div className="not-found-actions">
        <a className="not-found-primary" href="#/">
          Back to the library
        </a>
        <a className="not-found-secondary" href="#/models">
          Installed models
        </a>
      </div>
    </div>
  );
}
