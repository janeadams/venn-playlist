import logo from "./logo.svg";
import "./App.css";
import { useEffect, useState } from "react";
import { accessToken, logout, getCurrentUserProfile } from "./spotify";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useLocation,
} from "react-router-dom";

import Recommendations from "./components/Recommendations/Recommendations";
import Navbar from "./components/Navbar/Navbar";
import Login from "./components/Login/Login";
import Profile from "./components/Profile/Profile";
import TopArtists from "./components/TopArtists/TopArtists";
import TopTracks from "./components/TopTracks/TopTracks";
import Recent from "./components/Recent/Recent";
import Playlists from "./components/Playlists/Playlists";
import Artist from "./components/Artist/Artist";
import Playlist from "./components/Playlist/Playlist";
import Track from "./components/Track/Track";

function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}

function App() {
  const [token, setToken] = useState(null);

  useEffect(()=>{
    setToken(accessToken);
  },[])
  
  return (
    <Router>
      <div className="App">
        {console.log("got inside app")}
        {!token ? <Login /> : <Section />}
      </div>
    </Router>
  );
}

const Section = () => {
  return (
    <div className="App-container">
      {console.log("got inside app container")}
        <ScrollToTop />
        <Navbar />
        <Routes>
          <Route path="/top-artists" element={<TopArtists />} />
          <Route path="/top-tracks" element={<TopTracks />} />
          <Route path="/recent" element={<Recent />} />
          <Route path="/artist/:id" element={<Artist />} />
          <Route path="/playlist/:id" element={<Playlist />} />
          <Route path="/playlists" element={<Playlists />} />
          <Route path="/track/:id" element={<Track />} />
          <Route path="/recommendations/:id" element={<Recommendations />} />
          <Route path="/" element={<Profile logout={logout} />} />
        </Routes>
    </div>
  );
};

export default App;
