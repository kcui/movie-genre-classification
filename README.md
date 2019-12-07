# Deep Learning Final Project

Step 1: Initialize an environment from the given .yml by running 

`conda env create -f env.yml`

then activate with `conda activate movie-classification`.

Step 2: Run `python download-filmgrab.py` to scrape film images from filmgrab.com. The images are stored as `./filmgrab/<movie_name>/#.jpg` and a list of titles is stored in `data/movie-titles.txt`. Alternatively, download the prescraped dataset

https://www.dropbox.com/s/e1ip9pldog8mff8/filmgrab.tar.gz?dl=0&fbclid=IwAR3lA_WjcvMZYYUsPI9PcwrZq_wOPpdN2I6gpamtRZaSCSA_MqE6ob4FcjY

and run only `write_movie_titles()` in `main()`.

Step 3: Download and extract the IMDb movie info `title.basics.tsv` (with genre data) from https://datasets.imdbws.com/title.basics.tsv.gz and place in the root directory, then run `parse-imdb-data.py`. This will store a map from title to IMDb-assigned genre in `data/genre-map.txt` as a tsv and a map from movie frame path (relative) to genre (just one if `only_first_genre` is True, otherwise a comma-separated list of genres) in `data/frame-genre-map.txt`. This is the dataset we'll be using.
