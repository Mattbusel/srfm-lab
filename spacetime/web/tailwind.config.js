/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          base: '#0f0f14',
          card: '#1a1a24',
          elevated: '#22222f',
          border: '#2a2a3a',
        },
        accent: {
          DEFAULT: '#6366f1',
          hover: '#4f52e0',
          dim: '#6366f120',
        },
        bull: '#22c55e',
        bear: '#ef4444',
        sideways: '#9ca3af',
        highvol: '#a855f7',
        bh: {
          cold: '#6b7280',
          warm: '#eab308',
          hot: '#f97316',
          critical: '#ef4444',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 20px rgba(239,68,68,0.4)',
        'glow-yellow': '0 0 20px rgba(234,179,8,0.4)',
        'glow-accent': '0 0 20px rgba(99,102,241,0.4)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
    },
  },
  plugins: [],
}
