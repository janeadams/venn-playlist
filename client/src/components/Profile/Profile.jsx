import React, { useState, useEffect } from "react";
import "./Profile.css";
import { catchErrors } from "../../utils";
import {
  getCurrentUserPlaylists,
  getCurrentUserProfile,
  getFollowedArtists,
  getTopArtists,
  getTopTracks
} from "../../spotify";

import { logout } from "../../spotify";
import { formatDuration } from "../../utils";
import { Link } from "react-router-dom";
import {ReactComponent as InfoIcon} from '../../icons/info.svg';
import Loader from "../Loader";
import TrackList from "../TrackList/TrackList";

const Profile = () => {
  const [profile, setProfile] = useState(null);
  const [playlists, setPlaylists] = useState(null);
  const [following, setFollowing] = useState(null);
  const [topArtists, setTopArtists] = useState(null);
  const [topTracks, setTopTracks] = useState(null);


  useEffect(() => {
    const fetchData = async () => {
      const userProfile = await getCurrentUserProfile();
      setProfile(userProfile.data);

      const userPlaylists = await getCurrentUserPlaylists();
      setPlaylists(userPlaylists.data);

      const followedArtists = await getFollowedArtists();
      setFollowing(followedArtists.data);

      const userTopArtists = await getTopArtists(10);
      setTopArtists(userTopArtists.data);
        
      const userTopTracks = await getTopTracks(10);
      setTopTracks(userTopTracks.data);

    };
    

    catchErrors(fetchData());
  }, []);

  if (!profile || !playlists || !following || !topArtists || !topTracks) 
    return <Loader/>;

  return (
    <section className="profile">
      <div className="profile_header">
        <div className="profile_container">
          {profile.images.length && profile.images[0].url && (
            <img src={profile.images[0].url} alt="Avatar" />
          )}
          <a href={`https://open.spotify.com/user/${profile.id}`} target="_blank">
            <h1>{profile.display_name}</h1>
          </a>
        </div>
        <div className="stats">
          <div className="stat_element">
            <h3>{following.artists.total}</h3>
            <p>FOLLOWERS</p>
          </div>
          <div className="stat_element">
            <h3>{profile.followers.total}</h3>
            <p>FOLLOWING</p>
          </div>
          <div className="stat_element">
            <h3>{playlists.total}</h3>
            <p>PLAYLISTS</p>
          </div>
        </div>
        <button className="spotify_button" onClick={logout}>
          Logout
        </button>
      </div>
      <div className="tracks_preview">
        <div className="top_artists">
          <div className="top_preview_header">
            <h3>Top Artists of All Time</h3>
            <Link to="/top-artists" className="spotify_button">
              SEE MORE
            </Link>
          </div>
          <div className="preview_list_container">
            <ul className="artists_list preview_list">
              {topArtists.items.map((artist, idx) => (
                <Link to={`artist/${artist.id}`} className="artist" key={idx}>
                  <div className="preview_image_wrapper">
                    <img
                      className="preview_image"
                      src={artist.images[0].url}
                      alt=""
                    />
                    <div className="info_svg">
                      <InfoIcon />
                    </div>
                  </div>
                  <p>{artist.name}</p>
                </Link>
              ))}
            </ul>
          </div>
        </div>

        <div className="top_tracks">
          <div className="top_preview_header">
            <h3>Top Tracks of All Time</h3>
            <Link to="/top-artists" className="spotify_button">
              SEE MORE
            </Link>
          </div>
          <TrackList tracks={topTracks.items}/>
        </div>
      </div>
    </section>
  );
};

export default Profile;
