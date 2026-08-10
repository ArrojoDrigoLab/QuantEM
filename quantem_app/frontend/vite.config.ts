import { defineConfig, type UserConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

function parseBooleanEnv(value: string | undefined): boolean | null {
  if (value == null) {
    return null
  }
  const normalized = value.trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(normalized)) {
    return true
  }
  if (['0', 'false', 'no', 'off'].includes(normalized)) {
    return false
  }
  return null
}

function parsePositiveIntegerEnv(value: string | undefined): number | null {
  if (value == null || value.trim() === '') {
    return null
  }
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

const frontendRoot = path.resolve(__dirname)
const forcePollingDefault =
  frontendRoot.startsWith('/mnt/') || frontendRoot.startsWith('/workspace/')
const forcePolling =
  parseBooleanEnv(process.env.QUANTEM_VITE_USE_POLLING) ??
  parseBooleanEnv(process.env.CHOKIDAR_USEPOLLING) ??
  forcePollingDefault
const pollInterval =
  parsePositiveIntegerEnv(process.env.QUANTEM_VITE_POLL_INTERVAL_MS) ??
  parsePositiveIntegerEnv(process.env.CHOKIDAR_INTERVAL) ??
  300

type QuantEmViteConfig = UserConfig & {
  test: {
    environment: string
    setupFiles: string[]
    include: string[]
    exclude: string[]
    css: boolean
    coverage: {
      provider: string
      reporter: string[]
      include: string[]
      exclude: string[]
    }
  }
}

// https://vite.dev/config/
const config: QuantEmViteConfig = {
  // Relative base: the desktop shell serves the bundle from a custom app protocol
  // (or file://), where absolute "/assets/..." URLs do not resolve.
  base: './',
  plugins: [tailwindcss(), react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('/node_modules/')) {
            return undefined
          }
          // deck.gl, luma.gl and Viv are one mutually-recursive family. Splitting
          // them into separate chunks -- as this config used to -- puts a cycle
          // across a chunk boundary, and the app dies at load with
          // "ReferenceError: Cannot access 'Ai' before initialization" from the
          // luma chunk. The build does not catch it; only running it does.
          // They are always loaded together when the viewer opens, so a single
          // chunk costs nothing real.
          if (
            id.includes('/node_modules/@hms-dbmi/viv/') ||
            id.includes('/node_modules/@vivjs/') ||
            id.includes('/node_modules/@deck.gl/') ||
            id.includes('/node_modules/@luma.gl/') ||
            id.includes('/node_modules/@loaders.gl/')
          ) {
            return 'viewer-gl'
          }
          if (id.includes('/node_modules/geotiff/') || id.includes('/node_modules/ome-zarr/')) {
            return 'viewer-data'
          }
          if (id.includes('/blosc')) {
            return 'codec-blosc'
          }
          if (id.includes('/zstd')) {
            return 'codec-zstd'
          }
          return undefined
        },
      },
    },
  },
  resolve: {
    alias: {
      // Restrict app aliasing to "@/..." so scoped packages (e.g. "@deck.gl/*")
      // are always resolved from node_modules.
      '@/': `${path.resolve(__dirname, './src')}/`,
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    watch: forcePolling
      ? {
          usePolling: true,
          interval: pollInterval,
        }
      : undefined,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    exclude: ['src/**/._*'],
    css: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{ts,tsx}'],
      exclude: [
        'src/main.tsx',
        'src/**/*.old.tsx',
        'src/test/**',
        '**/._*',
      ],
    },
  },
}

export default defineConfig(config)
