/** One retained-tile allowance across base images, panes, overlays and generations. */
export class OverlayTileBudget {
  private owners = new Map<symbol, () => void>();
  readonly totalBytes: number;

  constructor(totalBytes = 128 * 1024 * 1024) { this.totalBytes = totalBytes; }

  register(owner: symbol, changed: () => void) {
    const added = !this.owners.has(owner);
    this.owners.set(owner, changed);
    if (added) this.notify();
  }

  release(owner: symbol) {
    if (this.owners.delete(owner)) this.notify();
  }

  get bytesPerLayer() {
    return Math.floor(this.totalBytes / Math.max(1, this.owners.size));
  }

  private notify() {
    for (const changed of this.owners.values()) changed();
  }
}

export const overlayTileBudget = new OverlayTileBudget();

type ImageTile = {
  content?: { width: number; height: number; data: ArrayBufferView | ArrayLike<{ byteLength: number }>; byteLength?: number } | null;
};

export function accountImageTile(tile: ImageTile) {
  const content = tile.content;
  if (!content) return;
  // Viv returns a single typed array for interleaved RGB(A), rather than a
  // list of channel arrays. Iterating its scalar values produces NaN bytes.
  if (ArrayBuffer.isView(content.data)) {
    content.byteLength = content.data.byteLength + content.width * content.height * 4;
    return;
  }
  let sourceBytes = 0;
  for (let index = 0; index < content.data.length; index++) sourceBytes += content.data[index].byteLength;
  // Reserve float textures even when the decoded channels are uint8/uint16.
  content.byteLength = sourceBytes + content.width * content.height * 4 * content.data.length;
}

/** Deck reads this immediately after onTileLoad, before cache accounting. */
export function accountOverlayTile(tile: ImageTile) {
  const content = tile.content;
  if (!content) return;
  if (ArrayBuffer.isView(content.data)) {
    content.byteLength = content.data.byteLength + content.width * content.height * 13;
    return;
  }
  let sourceBytes = 0;
  for (let index = 0; index < content.data.length; index++) sourceBytes += content.data[index].byteLength;
  // Reserve packed IDs, CPU color fallback and GPU label/border textures too.
  // Selected visible tiles can exceed Deck's cache allowance; this bounds
  // retention, rather than falsely claiming control of the driver's allocator.
  content.byteLength = sourceBytes + content.width * content.height * 13;
}
