import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  root: __dirname,
  server: {
    port: 5173,
    proxy: {
      '/events': 'http://localhost:3000',
      '/prevclose': 'http://localhost:3000',
      // Needed for spot change (diff/pct) API used by the dashboard
      '/spotdiff': 'http://localhost:3000'
    }
  },
  build: {
    outDir: 'dist'
  }
});
