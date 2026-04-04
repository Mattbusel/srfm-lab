/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        research: {
          bg: '#080b12',
          surface: '#0e1220',
          card: '#111827',
          border: '#1e2a3a',
          muted: '#2d3a4f',
          text: '#e2e8f0',
          subtle: '#8899aa',
          accent: '#3b82f6',
          'accent-dim': '#1d4ed8',
          bull: '#22c55e',
          bear: '#ef4444',
          sideways: '#6b7280',
          ranging: '#8b5cf6',
          volatile: '#f59e0b',
          warning: '#f59e0b',
          info: '#06b6d4',
          purple: '#8b5cf6',
          teal: '#14b8a6',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.15s ease-out',
        'slide-in': 'slideIn 0.2s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateY(-4px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
