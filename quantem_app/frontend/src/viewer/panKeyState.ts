/**
 * The one place that knows whether the space bar is being held to pan.
 *
 * Space has to do two jobs on the labeling canvas and they are both in the
 * plan: hold-and-drag is how you pan now that the left button belongs to the
 * active tool, and a tap is how you keep the object under the cursor without
 * moving the mouse to a sidebar button. A tap and the start of a drag are the
 * same keystroke, so the only thing that separates them is whether a pan
 * actually happened before the key came back up.
 *
 * That decision cannot live in either consumer alone -- the viewer knows a pan
 * happened, the labeling screen's keyboard hook decides whether to keep -- and
 * they are in different component trees with no shared ancestor. Space is a
 * global modifier the way Shift is, so it is tracked once here against
 * `window`, and both sides read the same state. `didSpacePan()` is set by the
 * pointer path, never by a key listener, so it does not depend on the order the
 * two `keyup` listeners happen to run in.
 */

type PanKeyListener = (held: boolean) => void;

let spaceHeld = false;
let spacePanned = false;
let listenerCount = 0;
const listeners = new Set<PanKeyListener>();

function isSpaceKey(event: KeyboardEvent): boolean {
  return event.key === " " || event.code === "Space";
}

function isTypingTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  return (
    element?.tagName === "INPUT" ||
    element?.tagName === "TEXTAREA" ||
    element?.isContentEditable === true
  );
}

function notify() {
  for (const listener of listeners) listener(spaceHeld);
}

function handleKeyDown(event: KeyboardEvent) {
  if (!isSpaceKey(event) || isTypingTarget(event.target)) return;
  if (spaceHeld) return;
  spaceHeld = true;
  spacePanned = false;
  notify();
}

function handleKeyUp(event: KeyboardEvent) {
  if (!isSpaceKey(event)) return;
  if (!spaceHeld) return;
  spaceHeld = false;
  notify();
}

function handleBlur() {
  if (!spaceHeld) return;
  spaceHeld = false;
  notify();
}

function subscribe(listener: PanKeyListener): () => void {
  listeners.add(listener);
  listenerCount += 1;
  if (listenerCount === 1) {
    window.addEventListener("keydown", handleKeyDown, true);
    window.addEventListener("keyup", handleKeyUp, true);
    window.addEventListener("blur", handleBlur);
  }
  return () => {
    if (!listeners.delete(listener)) return;
    listenerCount -= 1;
    if (listenerCount === 0) {
      window.removeEventListener("keydown", handleKeyDown, true);
      window.removeEventListener("keyup", handleKeyUp, true);
      window.removeEventListener("blur", handleBlur);
      spaceHeld = false;
      spacePanned = false;
    }
  };
}

/** Is the space bar down right now? */
export function isPanKeyHeld(): boolean {
  return spaceHeld;
}

/** The viewer calls this the moment a space-held drag actually moves the view. */
export function markSpacePan(): void {
  spacePanned = true;
}

/**
 * Did the space press that is ending now move the image?
 *
 * Reading it clears the flag, so one press can only ever suppress one keep.
 */
export function consumeSpacePan(): boolean {
  const panned = spacePanned;
  spacePanned = false;
  return panned;
}

/** Test seam: forget everything, as if no key had ever been pressed. */
export function resetPanKeyStateForTests(): void {
  spaceHeld = false;
  spacePanned = false;
}

export const panKeyState = {
  subscribe,
  isPanKeyHeld,
  markSpacePan,
  consumeSpacePan,
};
