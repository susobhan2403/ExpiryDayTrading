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
      '/prevclose': 'http://localhost:3000'
    }
  },
  build: {
    outDir: 'dist'
  }
});
