import React, { useEffect, useState } from "react";
import "./Artist.css";
import Loader from "../Loader";
import { catchErrors } from "../../utils";
import { getArtistInfo } from "../../spotify";
import { formatNumber } from "../../utils";

const Artist = () => {
  const url = window.location.pathname;
  const id = url.substring(url.lastIndexOf("/") + 1);
  const [artist, setArtist] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const artistInfo = await getArtistInfo(id);
      setArtist(artistInfo.data);
    };

    catchErrors(fetchData());
  }, []);

  
  if (!artist) return <Loader />;
  return (
    <section className="artist_container">
      <div className="artist_img">
        <img src={artist.images[0].url} alt="" />
        <a href={artist.external_urls.spotify} target="_blank" >
          <h1>{artist.name}</h1>
        </a>
      </div>
      <div className="artist_info">
        <div className="artist_followers">
          <div>{formatNumber(artist.followers.total)}</div>
          <p>FOLLOWERS</p>
        </div>
        <div className="artist_genres">
          <div>
            {artist.genres.map((genre, idx) => (
              <div className="" key={idx}>
                {genre}
              </div>
            ))}
          </div>
          <p>GENRES</p>
        </div>
        <div className="artist_popularity">
          <div>{artist.popularity}%</div>
          <p>POPULARITY</p>
        </div>
      </div>
    </section>
  );
};

export default Artist;
