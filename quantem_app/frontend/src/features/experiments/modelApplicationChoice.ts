import { persist, createJSONStorage } from "zustand/middleware";
import { create } from "zustand";

export interface ModelApplicationChoice {
  family: string;
  organelles: string[];
  adapter: string;
  threshold: number | null;
}

export const DEFAULT_APPLICATION_CHOICE: ModelApplicationChoice = {
  family: "auto", organelles: ["mito"], adapter: "", threshold: null,
};

export const useModelApplicationChoice = create<{
  choices: Record<string, ModelApplicationChoice>;
  update: (experiment: string, patch: Partial<ModelApplicationChoice>) => void;
}>()(persist((set) => ({
  choices: {},
  update: (experiment, patch) => set((state) => ({ choices: {
    ...state.choices,
    [experiment]: { ...(state.choices[experiment] ?? DEFAULT_APPLICATION_CHOICE), ...patch },
  } })),
}), { name: "quantem.modelApplication", storage: createJSONStorage(() => ({
  getItem(name) { try { return sessionStorage.getItem(name); } catch { return null; } },
  setItem(name, value) { try { sessionStorage.setItem(name, value); } catch { /* Keep the current session usable. */ } },
  removeItem(name) { try { sessionStorage.removeItem(name); } catch { /* Storage can be unavailable. */ } },
})) }));
