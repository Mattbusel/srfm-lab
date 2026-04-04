/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
    './node_modules/@tremor/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        dash: {
          bg: '#0a0b0e',
          surface: '#111318',
          border: '#1e2130',
          text: '#e2e8f0',
          subtle: '#94a3b8',
          muted: '#475569',
          accent: '#3b82f6',
          bull: '#22c55e',
          bear: '#ef4444',
          warning: '#f59e0b',
          info: '#06b6d4',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
