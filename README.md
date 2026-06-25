# Final Project for CSCI 4150 - Intro. To AI.
We are utilizing a lot of this github repository for the basic structure of the audio ingestion: https://github.com/Philippe-Guerrier/pulseflow-ai-offline-music-recommender 


The Presentation related to this github repo can be found here: {link}

# Running the tool:

To run the tool first run 
```
python main.py ingest
```
Then run
```
python -m scripts.recommend_from_playlist   --seed-dir "/Path/To/Playlist/You/Want/A/Recommendation/From"   --limit [number of songs to recommend]   --explain   --explain-limit [number of expainations to give
```

