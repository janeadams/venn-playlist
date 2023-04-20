import React from 'react'
import './Login.css'

const LOGIN_URI =
  process.env.NODE_ENV !== "production"
    ? "http://localhost:8888/login"
    : "https://spotify-profile-visualizer-app.herokuapp.com/login";
    
    
const Login = () => {
  return (
    <section className="login_section">
      <h1>Spotify Profile</h1>
      <a className="App-link" href={LOGIN_URI}>
        Log in to Spotify
      </a>
      {console.log("got inside login")}
    </section>
  );
}

export default Login