import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
})

// import { defineConfig } from 'vite';
// import react from '@vitejs/plugin-react';
// import fs from 'fs';
// import path from 'path';

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     https: {
//       key: fs.readFileSync(path.resolve(__dirname, 'C:/Users/jRuiz/Documents/projects/haystack/cert-key.pem')),
//       cert: fs.readFileSync(path.resolve(__dirname, 'C:/Users/jRuiz/Documents/projects/haystack/fullchain.pem'))
//     }
//   }
// });
