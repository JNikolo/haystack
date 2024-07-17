// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth, setPersistence } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCB4NLhMxUFE3PZhoyTrmk5KnxYgPoEbCs",
  authDomain: "haystack-b44fb.firebaseapp.com",
  projectId: "haystack-b44fb",
  storageBucket: "haystack-b44fb.appspot.com",
  messagingSenderId: "571379575383",
  appId: "1:571379575383:web:e7c1d9f2b4909533715112",
  measurementId: "G-JKZGS3V2S5"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const analytics = getAnalytics(app);

//setPersistence(auth, firebase.auth.Auth.Persistence.NONE);

export { auth, analytics };