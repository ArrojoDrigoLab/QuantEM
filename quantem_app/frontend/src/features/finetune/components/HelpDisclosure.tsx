/**
 * The "?" beside a section, and what it says.
 *
 * The owner asked for help "on hover". Hover alone is help that does not exist
 * for anyone using a keyboard or a touch screen, so this is a real button: it
 * opens on hover *and* on focus, it toggles on click and on Enter or Space
 * because it is a button, it closes on Escape, and it is wired to its panel
 * with `aria-expanded` and `aria-controls` so a screen reader is told there is
 * something to open. Hover is the shortcut, not the mechanism.
 */

import { useCallback, useEffect, useId, useState, type ReactNode } from "react";
import { cx } from "@/shared/ui/cx";

export function HelpDisclosure({
  label,
  title,
  children,
  className,
}: {
  /** What the button is *about*, for the accessible name: "training data". */
  label: string;
  title: string;
  children: ReactNode;
  className?: string;
}) {
  const panelId = useId();
  // Three independent reasons to be open, rather than one flag three handlers
  // fight over. With one flag, a click on a "?" the pointer is already hovering
  // toggles it *shut* -- the hover had already opened it -- so the one gesture
  // most people try does nothing visible.
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [pinned, setPinned] = useState(false);
  const open = hovered || focused || pinned;

  const closeAll = useCallback(() => {
    setHovered(false);
    setFocused(false);
    setPinned(false);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      // Stopping it here keeps Escape from also closing the dialog behind the
      // panel: one press, one thing closed.
      event.stopPropagation();
      closeAll();
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [open, closeAll]);

  return (
    <span
      className={cx("relative inline-flex", className)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={`About ${label}`}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-slate-300 bg-white text-xs font-semibold text-slate-600 hover:border-slate-400 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
        onClick={() => setPinned((current) => !current)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
      >
        ?
      </button>
      <span
        id={panelId}
        role="note"
        hidden={!open}
        className="absolute left-0 top-6 z-10 w-[26rem] max-w-[80vw] rounded-md border border-slate-200 bg-white p-3 text-left text-xs leading-relaxed text-slate-700 shadow-lg"
      >
        <span className="mb-1 block text-xs font-semibold text-slate-900">
          {title}
        </span>
        {children}
      </span>
    </span>
  );
}
