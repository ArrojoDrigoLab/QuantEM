/**
 * Every address the app answers, as data.
 *
 * The table used to be JSX inside `App.tsx`, and five separate packages needed
 * to register a route at once — a five-way conflict inside one `<Routes>`
 * block. As an array of homogeneous entries each of those is a one-line append
 * at the end, which merges cleanly where an edit inside JSX does not.
 *
 * **The rule for anyone adding a route:** add one `AppRoute` entry, at the end
 * of `APP_ROUTES`, and nothing else in this file. Order matters only for the
 * three entries that are already ordered against each other and say why.
 */

import { lazy, type ReactElement } from "react";
import { AssetRedirect, AssetRoute } from "./AssetRoute";
import { NotFoundScreen } from "./NotFoundScreen";

const LibraryPage = lazy(() =>
  import("@/features/library/LibraryPage").then((module) => ({
    default: module.LibraryPage,
  }))
);
const AnalysisScreen = lazy(() =>
  import("@/features/analysis/AnalysisScreen").then((module) => ({
    default: module.AnalysisScreen,
  }))
);
const ModelsScreen = lazy(() =>
  import("@/features/models/ModelsScreen").then((module) => ({
    default: module.ModelsScreen,
  }))
);

export interface AppRoute {
  /** The `path` prop of the `<Route>`. */
  path: string;
  /** The `element` prop of the `<Route>`. */
  element: ReactElement;
}

export const APP_ROUTES: AppRoute[] = [
  { path: "/", element: <LibraryPage /> },
  // Models are an app-level concern, not an image-level one: a user who never
  // opens Adapt still needs to know what is installed and what can run.
  { path: "/models", element: <ModelsScreen /> },
  { path: "/assets/:assetId", element: <AssetRedirect /> },
  // Declared before the generic :view route so "analysis" is never mistaken
  // for a viewer/labeling view name.
  { path: "/assets/:assetId/analysis", element: <AnalysisScreen /> },
  { path: "/assets/:assetId/:view/:segmentationTypeName?", element: <AssetRoute /> },
  // An address the router does not know says so. It used to redirect to the
  // library without a word, which recovers the session and hides the fact that
  // the link was wrong -- indistinguishable, from the reader's side, from the
  // app forgetting where they were. Keep this last.
  { path: "*", element: <NotFoundScreen /> },
];
