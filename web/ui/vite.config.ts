import { defineConfig } from 'vite'
import solid from 'vite-plugin-solid'

// `npm run dev` proxies the board API to web/server.py on :8000
export default defineConfig({
  plugins: [solid()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
