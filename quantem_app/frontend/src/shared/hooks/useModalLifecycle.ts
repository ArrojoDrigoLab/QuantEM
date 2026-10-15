import { useEffect, useRef, type RefObject } from "react";

type Modal = { element: HTMLElement };
const modals: Modal[] = [];
let previousOverflow = "";
const SELECTOR = 'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]';

/** One focus/keyboard contract, including a confirmation opened above a dialog. */
export function useModalLifecycle(
  panelRef: RefObject<HTMLElement | null>,
  onClose: () => void,
  { open = true, initialFocusRef }: { open?: boolean; initialFocusRef?: RefObject<HTMLElement | null> } = {},
) {
  const closeRef = useRef(onClose);
  useEffect(() => { closeRef.current = onClose; }, [onClose]);
  useEffect(() => {
    const panel = panelRef.current;
    if (!open || !panel) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const modal = { element: panel };
    if (!modals.length) {
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
    }
    const childIndex = modals.findIndex((entry) => panel.contains(entry.element));
    if (childIndex < 0) modals.push(modal);
    else modals.splice(childIndex, 0, modal);
    const focusable = () => [...panel.querySelectorAll<HTMLElement>(SELECTOR)].filter((element) =>
      element.tabIndex >= 0 && !element.closest('[hidden], [inert], [aria-hidden="true"]') &&
      getComputedStyle(element).display !== "none" && getComputedStyle(element).visibility !== "hidden",
    );
    if (!panel.hasAttribute("tabindex")) panel.setAttribute("tabindex", "-1");
    if (modals.at(-1) === modal) (initialFocusRef?.current ?? focusable()[0] ?? panel).focus();
    const handleKey = (event: KeyboardEvent) => {
      if (modals.at(-1) !== modal || event.defaultPrevented) return;
      if (event.key === "Escape") {
        event.preventDefault(); event.stopPropagation(); closeRef.current(); return;
      }
      if (event.key !== "Tab") return;
      const elements = focusable();
      const first = elements[0] ?? panel;
      const last = elements.at(-1) ?? panel;
      if (!elements.includes(document.activeElement as HTMLElement)) {
        event.preventDefault(); (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      const wasTop = modals.at(-1) === modal;
      modals.splice(modals.indexOf(modal), 1);
      document.removeEventListener("keydown", handleKey);
      if (!modals.length) document.body.style.overflow = previousOverflow;
      if (wasTop && previousFocus?.isConnected) previousFocus.focus();
    };
  }, [open, panelRef, initialFocusRef]);
}
