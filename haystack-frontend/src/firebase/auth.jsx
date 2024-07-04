import { auth } from "./config"; // Import Firebase auth instance
import { signInWithEmailAndPassword, createUserWithEmailAndPassword} from "firebase/auth";

export const signInWrapper = async (email, password) => {
    return signInWithEmailAndPassword(auth, email, password);
}

export const createUserWrapper = async (email, password) => {
    return createUserWithEmailAndPassword(auth, email, password);
}

export const signOutWrapper = async () => {
    return auth.signOut();
}