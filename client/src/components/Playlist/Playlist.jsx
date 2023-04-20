import React,{useState, useEffect} from 'react'
import './Playlist.css'
import { catchErrors } from '../../utils';
import { getPlaylistInfo } from '../../spotify';
import Loader from '../Loader';
import TrackList from '../TrackList/TrackList';
import { Link } from 'react-router-dom';
import FeatureChart from '../FeatureChart';
import { getAudioFeaturesForTracks } from '../../spotify';


const Playlist = () => {
    const url = window.location.pathname;
    const id = url.substring(url.lastIndexOf("/") + 1);
    const [playlist, setPlaylist] = useState(null);
    const [tracks, setTracks] = useState(null);

    const [audioFeatures, setAudioFeatures] = useState(null);


    useEffect(() => {
    const fetchData = async () => {
        const playlistInfo = await getPlaylistInfo(id);
        setPlaylist(playlistInfo.data);
        let playlist_tracks = []
        const tracks_arr = playlistInfo.data.tracks.items;
        for (var i = 0; i < tracks_arr.length;i++){
            playlist_tracks[i] = tracks_arr[i].track;
        }
        setTracks(playlist_tracks);
    };

    catchErrors(fetchData());
    }, []);

    useEffect(() => {
      const fetchData = async () => {
        if (playlist) {
          const { data } = await getAudioFeaturesForTracks(
            playlist.tracks.items
          );
          setAudioFeatures(data);
        }
      };
      catchErrors(fetchData());
    }, [playlist]);

    if (!playlist || !tracks || !audioFeatures) return <Loader />;
    
  return (
    <section className="playlist_section">
      <div className="playlist_left">
        <img src={playlist.images[0].url} alt="" />
        <h1>{playlist.name}</h1>
        <p className="playlist_owner">By {playlist.owner.display_name}</p>
        <p className="playlist_num_tracks">{playlist.tracks.total} tracks</p>
        <Link className="green_button" to={`/recommendations/${playlist.id}`}>GET RECOMMENDATIONS</Link>
        <div className="audio_features">
          <FeatureChart
            features={audioFeatures.audio_features}
            type="horizontalBar"
          />
        </div>
      </div>
      <div className="playlist_right">
        <TrackList tracks={tracks} />
      </div>
    </section>
  );
}

export default Playlist