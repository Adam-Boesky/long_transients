import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    // plotly.js reaches for Node's `global` in its cartesian modules, which
    // does not exist in a browser and which Vite does not polyfill. Without
    // this the bundle builds fine and then throws
    // `ReferenceError: global is not defined` at runtime.
    global: 'globalThis',
  },
  // `preview` serves the production build; it needs the same proxy as dev so
  // a built bundle can be exercised against the real API without CORS.
  preview: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  server: {
    proxy: {
      // Same-origin in dev, so the API needs no CORS entry for the dev port
      // and relative fetches work unchanged in production behind one host.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
