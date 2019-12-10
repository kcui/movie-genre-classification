import argparse
import csv
import unicodedata
import os

base_data_path = './data'
try:
    os.stat(base_data_path)
except:
    os.mkdir(base_data_path)

parser = argparse.ArgumentParser(description='IMDb data parser and mapper')
parser.add_argument('--imdb_data_url', type=str, help="url to scrape IMDb tsv information", default='https://datasets.imdbws.com/title.basics.tsv.gz', required=False)
parser.add_argument('-f', '--frames_path', type=str, help="input folder of movie frames", default="./film-grab", required=False)
args = parser.parse_args()

imdb_data_url = args.imdb_data_url
imdb_data_path = './title.basics.tsv'
movie_titles_path = './data/movie-titles.txt'
genre_map_path = './data/genre-map.txt'
frame_genre_map_path = './data/frame-genre-map.txt'

frames_path = args.frames_path

genre_map_dict = dict()

def format_title(title):
    illegal_chars = "'/:;,.!?&"
    for c in illegal_chars:
        title = title.replace(c, "")
    title = " ".join(title.strip().split()).replace(" ", "-")
    nfkd = unicodedata.normalize('NFKD', title)
    asciified = nfkd.encode('ASCII', 'ignore')
    return asciified.decode("ascii").lower()

def construct_genre_map():
    print("constructing genre map...")
    with open(imdb_data_path) as tsv, open(genre_map_path, 'w') as genre_map:
        titles = set(line.strip() for line in open(movie_titles_path))
        r = csv.DictReader(tsv, dialect='excel-tab')
        for row in r:
            # many titles have the original name appended to a translated name, or vice versa -- try those too
            potential_titles = [format_title(row['originalTitle']), format_title(row['primaryTitle']), format_title(row['originalTitle']) + "-" + format_title(row['primaryTitle']), format_title(row['primaryTitle']) + "-" + format_title(row['originalTitle'])]
            for potential_title in potential_titles:
                if potential_title in titles:
                    titles.remove(potential_title)
                    if row['genres'] != "\\N":
                        genre_map.write("%s\t%s\n" % (potential_title, row['genres']))
                        genre_map_dict[potential_title] = row['genres']

        # print(len(titles))
        print("warning: unable to get genre information for %d titles; these will be skipped" % len(titles))
    print("finished constructing genre map (stored in data/genre-map.txt).")

def populate_genre_map_dict(only_first_genre=False):
    print("populating genre dict...")
    try:
        with open(genre_map_path) as map:
            for line in map:
                movie_title, genres = line.split('\t')
                if only_first_genre:
                    genre_map_dict[movie_title] = genres.split(',')[0].strip()
                else:
                    genre_map_dict[movie_title] = genres.strip()
        
        print("finished populating genre dict.")
    except:
        print("error: genre-map.txt could not be found; aborting")


def construct_frame_genre_map(only_first_genre=False):
    """
    Writes a text file of frame path to genre.
    """
    if len(genre_map_dict) == 0:
        populate_genre_map_dict()
    print("constructing frame genre map...")
    with open(frame_genre_map_path, 'w') as map:
        for (dirpath, _, filenames) in os.walk(frames_path):
            for filename in filenames:
                movie_title = os.path.basename(dirpath)
                if movie_title in genre_map_dict:
                    map.write("%s\t%s\n" % (os.path.join(dirpath, filename), genre_map_dict[movie_title]))
    print("done constructing frame genre map (stored in data/frame-genre-map.txt).")
    


if __name__ == "__main__":
    pass
    # UNCOMMENT TO RUN
    try:
        os.stat(genre_map_path)
        print("genre map exists already!")
    except:
        construct_genre_map()
    construct_frame_genre_map()