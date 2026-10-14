import { defineConfig } from 'vite';
export default defineConfig({ build: { lib: { entry: 'index.ts', formats: ['es'], fileName: 'g4-leto' } } });
