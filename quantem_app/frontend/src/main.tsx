import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import './index.css'
import App from './app/App.tsx'

// HashRouter, not BrowserRouter: the desktop shell serves the bundle from a custom
// app protocol / file:// origin where a path-based route 404s on reload.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
)
