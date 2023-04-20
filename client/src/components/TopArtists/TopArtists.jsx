import React from "react";
import "./TopArtists.css";
import { getTopArtists } from "../../spotify";
import { catchErrors } from "../../utils";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ReactComponent as InfoIcon } from "../../icons/info.svg";
import Loader from "../Loader";

const TopArtists = () => {
  const [topArtists, setTopArtists] = useState(null);
  const [activeRange, setActiveRange] = useState("long_term");

  useEffect(() => {
    const fetchData = async () => {
      const userTopArtists = await getTopArtists(20, activeRange);
      setTopArtists(userTopArtists.data);
    };

    catchErrors(fetchData());
  }, [activeRange]);

  if (!topArtists) return <Loader />;

  return (
    <section className="top_artists_section">
      <div className="top_header">
        <h1>Top Artists</h1>
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
      <div className="top_artists_wrapper">
        {topArtists.items.map((artist, idx) => (
          <div className="artist_wrapper" key={idx}>
            <Link to={`/artist/${artist.id}`} className="artist_img_wrapper">
              <img src={artist.images[0].url} alt="" />
              <div className="info_svg">
                <InfoIcon />
              </div>
            </Link>
            <h1>{artist.name}</h1>
          </div>
        ))}
      </div>
    </section>
  );
};

export default TopArtists;
