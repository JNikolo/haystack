import { auth } from "./config"; // Import Firebase auth instance
import { signInWithEmailAndPassword, createUserWithEmailAndPassword} from "firebase/auth";

export const signInWrapper = async (email, password) => {
    return signInWithEmailAndPassword(auth, email, password)//.then(user => {
        // // Get the user's ID token as it is needed to exchange for a session cookie.
        // return user.getIdToken().then(idToken => {
        //     // Session login endpoint is queried and the session cookie is set.
        //     // CSRF protection should be taken into account.
        //     // ...
        //     const csrfToken = getCookie('csrfToken')
        //     return postIdTokenToSessionLogin('http://127.0.0.1:8000/sessionLogin', idToken, csrfToken);
        // });
        // }).then(() => {
        // // A page redirect would suffice as the persistence is set to NONE.
        // return auth.signOut();
        // })
}

export const createUserWrapper = async (email, password) => {
    return createUserWithEmailAndPassword(auth, email, password);
}

export const signOutWrapper = async () => {
    return auth.signOut();
}


async function postIdTokenToSessionLogin(url, idToken, csrfToken) {
    // POST to session login endpoint.
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ idToken: idToken, csrfToken: csrfToken})
    });

    return response.json();
}