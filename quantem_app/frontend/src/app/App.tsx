/**
 * Main App component that routes between the library, the viewer and the
 * labeling (segmentation) screen.
 *
 * The table it renders is `app/routes.tsx`; the asset routes' own logic is
 * `app/AssetRoute.tsx`. Both were split out of this file so that a package
 * registering a route appends one array entry instead of editing JSX five ways
 * at once.
 */

import { Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { RouteFallback } from "./AssetRoute";
import { APP_ROUTES } from "./routes";
import "./App.css";

function App() {
  return (
    <div className="app">
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          {APP_ROUTES.map((route) => (
            <Route key={route.path} path={route.path} element={route.element} />
          ))}
        </Routes>
      </Suspense>
    </div>
  );
}

export default App;
