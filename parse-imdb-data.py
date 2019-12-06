import csv
import unicodedata
import os

base_data_path = './data'
try:
    os.stat(base_data_path)
except:
    os.mkdir(base_data_path)

imdb_data_url = 'https://datasets.imdbws.com/title.basics.tsv.gz'
imdb_data_path = './title.basics.tsv'
movie_titles_path = './data/movie-titles.txt'
genre_map_path = './data/genre-map.txt'

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

        # print(len(titles))
        print("warning: unable to get genre information for %d titles; these will be skipped" % len(titles))
    print("finished constructing genre map (stored in data/genre-map.txt).")

if __name__ == "__main__":
    pass
    # UNCOMMENT TO RUN
    construct_genre_map()