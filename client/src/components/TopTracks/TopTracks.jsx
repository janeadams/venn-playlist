import React from "react";
import "./TopTracks.css";
import { useState,useEffect } from "react";
import { getTopTracks } from "../../spotify";
import { catchErrors } from "../../utils";
import { Link } from "react-router-dom";
import { ReactComponent as InfoIcon } from "../../icons/info.svg";
import { formatDuration } from "../../utils";
import {default as Loader} from "../Loader";
import TrackList from "../TrackList/TrackList";

const TopTracks = () => {
    const [topTracks, setTopTracks] = useState(null);
    const [activeRange, setActiveRange] = useState("long_term");

    useEffect(() => {
      const fetchData = async () => {
        const userTopTracks = await getTopTracks(20, activeRange);
        setTopTracks(userTopTracks.data);
      };


      catchErrors(fetchData());
    }, [activeRange]);

    if (!topTracks) return <Loader/>;
  return (
    <section className="top_tracks_section">
      <div className="top_header">
        <h1>Top Tracks</h1>
        <div className="top_range">
          <div
            className={activeRange === "long_term" ? "active" : ""}
            onClick={() => setActiveRange("long_term")}
          >
            All Time
          </div>
          <div
            className={activeRange === "medium_term" ? "active" : ""}
            onClick={() => setActiveRange("medium_term")}
          >
            Last 6 Months
          </div>
          <div
            className={activeRange === "short_term" ? "active" : ""}
            onClick={() => setActiveRange("short_term")}
          >
            Last 4 Weeks
          </div>
        </div>
      </div>
      <TrackList tracks={topTracks.items}/>
    </section>
  );
};

export default TopTracks;
