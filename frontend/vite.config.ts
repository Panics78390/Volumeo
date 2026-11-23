import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// En développement, les appels /api sont relayés vers le backend FastAPI.
const backend = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: { '/api': { target: backend, changeOrigin: true } },
  },
  build: { chunkSizeWarningLimit: 1500 },
})
