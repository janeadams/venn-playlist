import React from 'react'
import { getRecentlyPlayedTracks } from '../../spotify';
import './Recent.css'
import { useState,useEffect } from 'react';
import { catchErrors } from '../../utils';
import { ReactComponent as InfoIcon } from "../../icons/info.svg";
import { formatDuration } from '../../utils';
import { Link } from 'react-router-dom';
import Loader from '../Loader';

const Recent = () => {
      const [recentTracks, setRecentTracks] = useState(null);

      useEffect(() => {
        const fetchData = async () => {
          const userRecentTracks = await getRecentlyPlayedTracks();
          setRecentTracks(userRecentTracks.data);
        };

        catchErrors(fetchData());
      }, []);

      if (!recentTracks) return <Loader />;

  return (
    <section className="recent_section">
      <div className="top_header">
        <h1>Recently Played Tracks</h1>
      </div>
      <div className="preview_list_container">
        <ul className="preview_list">
          {recentTracks.items.map((track, idx) => (
            <Link to={`/track/${track.track.id}`} key={idx}>
              <div className="preview_image_wrapper">
                <img
                  className="preview_image track"
                  src={track.track.album.images[0].url}
                  alt=""
                />
                <div className="info_svg">
                  <InfoIcon />
                </div>
              </div>

              <div className="track_info">
                <div className="track_details">
                  <p>{track.track.name}</p>
                  <span className="track_artists">
                    {track.track.artists.map((artist, i) => (
                      <span key={i}>
                        {artist.name}
                        {i !== track.track.artists.length - 1 && ", "}
                      </span>
                    ))}
                    {"  ·  "}
                    <span>{track.track.album.name}</span>
                  </span>
                </div>
                <span className="track_time">
                  {formatDuration(track.track.duration_ms)}
                </span>
              </div>
            </Link>
          ))}
        </ul>
      </div>
    </section>
  );
}

export default Recent