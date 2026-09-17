import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// FastAPI serves the built web/dist in production; in dev this proxy makes
// the same relative /answer and /health URLs work against a local API run.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/answer': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
