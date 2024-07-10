import { auth } from "./config"; // Import Firebase auth instance
import { signInWithEmailAndPassword, createUserWithEmailAndPassword} from "firebase/auth";

export const signInWrapper = async (email, password) => {
    try{
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user= userCredential.user;

        const idToken= await user.getIdToken();

        //const csrfToken = getCookie('csrfToken');

        await postIdTokenToSessionLogin('http://127.0.0.1:8000/session_login', idToken);//, csrfToken);

    // window.localStorage.setItem("isLogged", true);
        
        await auth.signOut();
    }catch (error) {
        console.error('Error signing in:', error);
        throw error; // Handle error appropriately in your application
    }
}

export const createUserWrapper = async (email, password) => {
    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        await signInWrapper(email, password);
    } catch (error) {
        console.error('Error creating user:', error);
        throw error; // Handle error appropriately in your application
    }
}

export const signOutWrapper = async () => {
    try {
        await auth.signOut();
    } catch (error) {
        console.error('Error signing out:', error);
        throw error; // Handle error appropriately in your application
    }
}


async function postIdTokenToSessionLogin(url, idToken){//, csrfToken) {
    // POST to session login endpoint.
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            mode: 'cors', 
            credentials: 'include',
            body: JSON.stringify({ idToken: idToken })//, csrfToken: csrfToken })
        });

        if (!response.ok) {
            throw new Error('Failed to post ID token to session login endpoint');
        }

        return response.json();
    } catch (error) {
        console.error('Error posting ID token to session login:', error);
        throw error; // Handle error appropriately in your application
    }
}

function getCookie(name) {
    const cookieValue = document.cookie.match('(^|[^;]+)\\s*' + name + '\\s*=\\s*([^;]+)');
    return cookieValue ? cookieValue.pop() : '';
}

export async function verifyLogin(){
    try {
        const response = await fetch("http://127.0.0.1:8000/verify", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            mode: 'cors', 
            credentials: 'include',
        });
        if (response.ok) {
            return true;
        } else {
            return false;
        }
    } catch (error) {
        console.error('Error verifying login:', error)
        return false;
    }
}
