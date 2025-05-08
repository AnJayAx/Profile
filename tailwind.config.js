/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.{html,js}",
    "./static/**/*.{js,css}"
  ],
  theme: {
    extend: {
      animation: {
        'fadeIn': 'fadeIn 0.5s ease-out forwards',
        'pulse': 'pulse 1.2s infinite',
        'typewriter': 'typewriter 2s steps(40) forwards',
        'typewriter-slow': 'typewriter 3.5s steps(60) forwards',
        'cursor-blink': 'cursor-blink 0.8s infinite'
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        },
        pulse: {
          '0%, 100%': { opacity: '0.6' },
          '50%': { opacity: '1' }
        },
        typewriter: {
          '0%': { width: '0%' },
          '100%': { width: '100%' }
        },
        'cursor-blink': {
          '0%, 100%': { borderRightColor: 'transparent' },
          '50%': { borderRightColor: 'white' }
        }
      }
    },
  },
  plugins: [],
}
