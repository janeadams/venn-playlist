import React, { useMemo } from "react";
import "./Recommendations.css";
import { Link } from "react-router-dom";
import { getPlaylistInfo } from "../../spotify";
import Loader from "../Loader";
import { useState, useEffect } from "react";
import { catchErrors } from "../../utils";
import { getRecommendationsForTracks } from "../../spotify";
import { getCurrentUserProfile } from "../../spotify";
import TrackList from "../TrackList/TrackList";
import { createPlaylist } from "../../spotify";
import { doesUserFollowPlaylist } from "../../spotify";

import { addTracksToPlaylist } from "../../spotify";  
import { followPlaylist } from "../../spotify";

const Recommendations = () => {
  const url = window.location.pathname;
  const playlistID = url.substring(url.lastIndexOf("/") + 1);

  const [playlist, setPlaylist] = useState(null);
  const [userID, setUserID] = useState(null);
  const [recommendations, setRecommmendations] = useState(null);
  const [recPlaylistId, setRecPlaylistId] = useState(null);
  const [isFollowing, setIsFollowing] = useState(null);

  useEffect(() => {
    const fetchPlaylistData = async () => {
      const { data } = await getPlaylistInfo(playlistID);
      setPlaylist(data);
    };
    catchErrors(fetchPlaylistData());

    const fetchUserData = async () => {
      const { data } = await getCurrentUserProfile();
      setUserID(data.id);
    };
    catchErrors(fetchUserData());
  }, [playlistID]);

  useMemo(() => {
    const fetchData = async () => {
      if (playlist) {
        const { data } = await getRecommendationsForTracks(
          playlist.tracks.items
        );
        setRecommmendations(data);
      }
    };
    catchErrors(fetchData());
  }, [playlist]);

  // If recPlaylistId has been set, add tracks to playlist and follow
  useMemo(() => {
    const isUserFollowingPlaylist = async (plistId) => {
      const { data } = await doesUserFollowPlaylist(plistId, userID);
      setIsFollowing(data[0]);
    };

    const addTracksAndFollow = async () => {
      const uris = recommendations.tracks.map(({ uri }) => uri).join(",");
      const { data } = await addTracksToPlaylist(recPlaylistId, uris);

      // Then follow playlist
      if (data) {
        await followPlaylist(recPlaylistId);
        // Check if user is following so we can change the save to spotify button to open on spotify
        catchErrors(isUserFollowingPlaylist(recPlaylistId));
      }
    };

    if (recPlaylistId && recommendations && userID) {
      catchErrors(addTracksAndFollow(recPlaylistId));
    }
  }, [recPlaylistId, recommendations, userID]);

  const createPlaylistOnSave = async () => {
    if (!userID) {
      return;
    }

    const name = `Recommended Tracks Based on ${playlist.name}`;
    const { data } = await createPlaylist(userID, name);
    setRecPlaylistId(data.id);
  };

  if (!recommendations) return <Loader />;

  return (
    <section className="recommendations_section">
      <div className="top_header recommend_header">
        <h1>
          Recommended Tracks Based on <span>{playlist.name}</span>
        </h1>
        {isFollowing && recPlaylistId ? (
          <a
            className="white_button"
            href={`https://open.spotify.com/playlist/${recPlaylistId}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            Open in Spotify
          </a>
        ) : (
          <a onClick={catchErrors(createPlaylistOnSave)} className="green_button">
            SAVE TO SPOTIFY
          </a>
        )}
      </div>
      <div className="subtitle_wrapper">
        <p>Refresh to get new tracks.</p>
      </div>
      <div className="recommended_tracks">
        <TrackList tracks={recommendations.tracks} />
      </div>
    </section>
  );
};

export default Recommendations;
