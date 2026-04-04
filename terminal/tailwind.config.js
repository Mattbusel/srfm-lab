/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: '#0a0e17',
          surface: '#111827',
          border: '#1f2937',
          muted: '#374151',
          text: '#e5e7eb',
          subtle: '#9ca3af',
          accent: '#3b82f6',
          'accent-dim': '#1d4ed8',
          bull: '#22c55e',
          bear: '#ef4444',
          sideways: '#6b7280',
          warning: '#f59e0b',
          info: '#06b6d4',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'flash-green': 'flashGreen 0.4s ease-out',
        'flash-red': 'flashRed 0.4s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in': 'slideIn 0.2s ease-out',
        'fade-in': 'fadeIn 0.15s ease-out',
      },
      keyframes: {
        flashGreen: {
          '0%': { backgroundColor: 'rgba(34, 197, 94, 0.3)' },
          '100%': { backgroundColor: 'transparent' },
        },
        flashRed: {
          '0%': { backgroundColor: 'rgba(239, 68, 68, 0.3)' },
          '100%': { backgroundColor: 'transparent' },
        },
        slideIn: {
          '0%': { transform: 'translateY(-4px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      gridTemplateColumns: {
        terminal: '280px 1fr 320px',
        'terminal-wide': '320px 1fr 360px',
        'terminal-compact': '240px 1fr 280px',
      },
    },
  },
  plugins: [],
}
