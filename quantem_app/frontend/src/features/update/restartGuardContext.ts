import { createContext } from "react";

export interface RestartBlocker {
  id: string;
  message: string;
}

export interface RestartGuardValue {
  blockers: RestartBlocker[];
  setBlocker: (id: string, message: string | null) => void;
}

export const RestartGuardContext = createContext<RestartGuardValue | null>(null);
