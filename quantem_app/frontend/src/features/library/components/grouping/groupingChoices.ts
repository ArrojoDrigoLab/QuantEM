/** The grouping intents shared by import and bulk-selection controls. */
export type GroupingChoice =
  | { kind: "keep" }
  | { kind: "none" }
  | { kind: "existing"; id: string }
  | { kind: "new"; name: string };

export const NO_GROUP: GroupingChoice = { kind: "none" };
export const KEEP_GROUP: GroupingChoice = { kind: "keep" };

export function chosenId(choice: GroupingChoice): string {
  return choice.kind === "existing" ? choice.id : "";
}

export function chosenName(choice: GroupingChoice): string {
  return choice.kind === "new" ? choice.name.trim() : "";
}

export function isChosen(choice: GroupingChoice): boolean {
  return Boolean(chosenId(choice) || chosenName(choice));
}
