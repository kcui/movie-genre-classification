import argparse
import io, os, re
import unicodedata
import zipfile
from bs4 import BeautifulSoup
from multiprocessing import Pool
from urllib.parse import quote
from urllib.request import urlopen

parser = argparse.ArgumentParser(description='Filmgrab scraper')
parser.add_argument('-o', '--output', type=str, help="destination", default="./film-grab", required=False)

args = parser.parse_args()

BASE = "https://film-grab.com/"

dir_path = os.getcwd()
download_path = os.path.normpath(os.path.join(dir_path, args.output))

try:
    os.stat(download_path)
except:
    os.mkdir(download_path)  

def asciify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def get_movie_urls():
    movie_list_html = urlopen(BASE + "movies-a-z/").read()
    soup = BeautifulSoup(movie_list_html, "lxml")
    movie_items = soup.find("ul", "display-posts-listing")
    movie_urls = [movie.a["href"] for movie in movie_items.findAll("li")]
    return movie_urls


def get_movie_frames(movie_url, movie_num):
    movie_html = urlopen(movie_url).read()
    soup = BeautifulSoup(movie_html, "lxml")

    movie_title = asciify(soup.title.string.split(" – FILMGRAB", 1)[0])

    print("downloading movie %d: %s" % (movie_num, movie_title))

    movie_download_folder = os.path.join(download_path, movie_title)
    try:
        os.stat(movie_download_folder)
    except:
        os.mkdir(movie_download_folder)  

    movie_images_zip_div = soup.find("div", "bwg_download_gallery")
    if movie_images_zip_div is not None:
        # get download button click url
        movie_images_zip_href = movie_images_zip_div.a["href"]
        response = urlopen(movie_images_zip_href)
        with zipfile.ZipFile(io.BytesIO(response.read())) as zipf:
            frame_count = 0
            for file in zipf.namelist():
                zipf.extract(file, path = movie_download_folder)
                _, extension = os.path.splitext(file)
                new_filename = str(frame_count) + extension
                os.rename(os.path.join(movie_download_folder, file), os.path.join(movie_download_folder, new_filename))
                frame_count += 1
    else:
        # fallback onto scraping raw images from page
        movie_images_container_div = soup.find("div", "bwg-container-0")
        if movie_images_container_div is not None:
            movie_image_urls = [image_container_div.a["href"] for image_container_div in movie_images_container_div.findAll("div", "bwg-item")]
            frame_count = 0
            for image_url in movie_image_urls:
                image_open_url = urlopen(quote(image_url, safe="/:()?="))
                image_open_url_headers = image_open_url.headers
                contentType = image_open_url_headers.get('content-type')
                image_data = image_open_url.read()

                f = open(os.path.join(movie_download_folder, str(frame_count) + ".jpg"), 'wb')
                f.write(image_data)
                f.close()

                frame_count += 1
        else:
            # case where there is no download and no data; just print error
            print("WARNING: movie %d (%s) has no scrapable frames" % (movie_num, movie_title))

movie_urls = get_movie_urls()
for movie_num, movie_url in enumerate(movie_urls[1:]):
    movie_image_urls = get_movie_frames(movie_url, movie_num+1)
    