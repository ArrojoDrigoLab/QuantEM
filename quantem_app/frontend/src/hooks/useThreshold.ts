/**
 * Hook for managing threshold state with debouncing.
 */

import { useState, useEffect } from "react";

export function useThreshold(initialValue: number = 0.99) {
  const [threshold, setThreshold] = useState(initialValue);
  const [debouncedThreshold, setDebouncedThreshold] = useState(initialValue);

  useEffect(() => {
    const id = setTimeout(() => setDebouncedThreshold(threshold), 250);
    return () => clearTimeout(id);
  }, [threshold]);

  return {
    threshold,
    debouncedThreshold,
    setThreshold,
  };
}
