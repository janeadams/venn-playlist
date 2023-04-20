import React from "react";
import { Link } from "react-router-dom";
import { ReactComponent as InfoIcon } from "../../icons/info.svg";
import { formatDuration } from "../../utils";
import './TrackList.css'

const TrackList = ({tracks}) => {
  return (
      <div className="preview_list_container track_list">
        <ul className="preview_list">
          {tracks.map((track, idx) => (
            <Link to={`/track/${track.id}`} key={idx}>
              <div className="preview_image_wrapper">
                <img
                  className="preview_image track"
                  src={track.album.images[0].url}
                  alt=""
                />
                <div className="info_svg">
                  <InfoIcon />
                </div>
              </div>

              <div className="track_info">
                <div className="track_details">
                  <p>{track.name}</p>
                  <span className="track_artists">
                    {track.artists.map((artist, i) => (
                      <span key={i}>
                        {artist.name}
                        {i !== track.artists.length - 1 && ", "}
                      </span>
                    ))}
                    {"  ·  "}
                    <span>{track.album.name}</span>
                  </span>
                </div>
                <span className="track_time">
                  {formatDuration(track.duration_ms)}
                </span>
              </div>
            </Link>
          ))}
        </ul>
      </div>
  );
};

export default TrackList;
