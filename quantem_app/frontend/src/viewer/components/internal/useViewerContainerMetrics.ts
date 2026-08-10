import { useEffect, useRef, useState } from "react";

export function useViewerContainerMetrics() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setContainerSize({ width: Math.max(1, width), height: Math.max(1, height) });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return { containerRef, containerSize };
}

