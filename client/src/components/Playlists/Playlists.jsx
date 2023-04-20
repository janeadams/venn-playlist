import React from "react";
import { getCurrentUserPlaylists } from "../../spotify";
import "./Playlists.css";
import { useState, useEffect } from "react";
import { catchErrors } from "../../utils";
import { Link } from "react-router-dom";
import Loader from "../Loader";

const Playlists = () => {
  const [playlists, setPlaylists] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const userPlaylists = await getCurrentUserPlaylists();
      setPlaylists(userPlaylists.data);
    };

    catchErrors(fetchData());
  }, []);

  if (!playlists) return <Loader />;

  return (
    <section className="playlists_section">
      <div className="top_header">
        <h1>Your Playlists</h1>
      </div>
      <div className="playlists_wrapper">
        {playlists.items.map((playlist, idx) => (
          <div className="playlist" key={idx}>
            <Link to={`/playlist/${playlist.id}`}>
              {playlist.images[0] && (
                <img src={playlist.images[0].url} alt="" />
              )}
              <div></div>
            </Link>
            <div className="playlist_details">
              <Link to={`/playlist/${playlist.id}`}>{playlist.name}</Link>
              <p>{playlist.tracks.total} tracks</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Playlists;
