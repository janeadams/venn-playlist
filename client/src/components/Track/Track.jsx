import React from "react";
import { useState, useEffect } from "react";
import { catchErrors, getYear } from "../../utils";
import Loader from "../Loader";
import { getTrackInfo } from "../../spotify";
import './Track.css'
import { formatDuration } from "../../utils";
import { parsePitchClass } from "../../utils";
import FeatureChart from "../FeatureChart";

const Track = () => {
  const url = window.location.pathname;
  const trackID = url.substring(url.lastIndexOf("/") + 1);

  const [track, setTrack] = useState(null);
  const [audioAnalysis, setAudioAnalysis] = useState(null);
  const [audioFeatures, setAudioFeatures] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const data = await getTrackInfo(trackID);
      setTrack(data.track);
      setAudioAnalysis(data.audioAnalysis);
      setAudioFeatures(data.audioFeatures);
    };
    catchErrors(fetchData());
  }, [trackID]);

  if (!track || !audioAnalysis || !audioFeatures) return <Loader />;
  return (
    <section className="track_section">
      <div className="track_data_wrapper">
        <img src={track.album.images[0].url} alt="" />
        <div className="track_information">
          <h1>{track.name}</h1>
          <h2>
            {track.artists &&
              track.artists.map(({ name }, i) => (
                <span key={i}>
                  {name}
                  {track.artists.length > 0 && i === track.artists.length - 1
                    ? ""
                    : ","}
                  &nbsp;
                </span>
              ))}
          </h2>
          <p className="album_text">
            <a
              href={track.album.external_urls.spotify}
              target="_blank"
              rel="noopener noreferrer"
            >
              {track.album.name}
            </a>{" "}
            &middot; {getYear(track.album.release_date)}
          </p>
          <a
            className="green_button"
            href={track.external_urls.spotify}
            target="_blank"
            rel="noopener noreferrer"
          >
            Play on Spotify
          </a>
        </div>
      </div>
      <div className="track_audio_wrapper">
        <div className="audio_features">
          <div className="feature_grid">
            <div>
              <h3>{formatDuration(audioFeatures.duration_ms)}</h3>
              <p>Duration</p>
            </div>
            <div>
              <h3>{parsePitchClass(audioFeatures.key)}</h3>
              <p>Key</p>
            </div>
            <div>
              <h3>{audioFeatures.mode === 1 ? "Major" : "Minor"}</h3>
              <p>Modality</p>
            </div>
            <div>
              <h3>{audioFeatures.time_signature}</h3>
              <p>Time Signature</p>
            </div>
            <div>
              <h3>{Math.round(audioFeatures.tempo)}</h3>
              <p>Tempo (BPM)</p>
            </div>
            <div>
              <h3>{track.popularity}%</h3>
              <p>Popularity</p>
            </div>
            <div>
              <h3>{audioAnalysis.bars.length}</h3>
              <p>Bars</p>
            </div>
            <div>
              <h3>{audioAnalysis.beats.length}</h3>
              <p>Beats</p>
            </div>
            <div>
              <h3>{audioAnalysis.sections.length}</h3>
              <p>Sections</p>
            </div>
            <div>
              <h3>{audioAnalysis.segments.length}</h3>
              <p>Segments</p>
            </div>
          </div>
          <div className="feature_chart">
            <FeatureChart features={audioFeatures} type="" />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Track;
